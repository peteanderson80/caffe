#include <vector>
#include <cfloat>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>

#include "caffe/layers/beam_search_node_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
void BeamSearchNodeLayer<Dtype>::reverse_gpu_sort(
        Blob<Dtype>& score, Blob<int>& index, const int chunksize){
  Dtype* score_data = score.mutable_gpu_data();
  int* index_data = index.mutable_gpu_data();
  // Make temp usage of diff to sort every 'chunksize' of scores separately
  int* idx_index_data = index.mutable_gpu_diff();
  thrust::device_ptr<Dtype> score_ptr = thrust::device_pointer_cast(score_data);
  thrust::device_ptr<int> index_ptr = thrust::device_pointer_cast(index_data);
  thrust::device_ptr<int> idx_index_ptr = thrust::device_pointer_cast(idx_index_data);
  // Initialize idx_index_data to give each block of data an index
  thrust::transform(thrust::make_counting_iterator(0),
              thrust::make_counting_iterator(score.count()),
              thrust::make_constant_iterator(chunksize),
              idx_index_ptr,
              thrust::divides<int>());
  // stable sort - greatest score first   
  thrust::stable_sort_by_key(score_ptr, score_ptr + score.count(),
              thrust::make_zip_iterator(thrust::make_tuple(idx_index_ptr, index_ptr)),
              thrust::greater<Dtype>());        
  // stable sort - lowest data block first
  thrust::stable_sort_by_key(idx_index_ptr, idx_index_ptr + score.count(),
              thrust::make_zip_iterator(thrust::make_tuple(score_ptr, index_ptr)));
  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
__global__ void SumScores(
  const int nthreads,
  const int timestep,
  const Dtype end_of_sequence,
  const int beam_size,
  const int vocab_size,
  const bool prevent_repeats,
  const int allowed_multiple_size,
  const Dtype* allowed_multiple_data,
  const Dtype* seq,
  const Dtype* output_score_sorted,
  const int* output_score_indices_data,
  const Dtype* partial_score,
  Dtype* score_data,
  int* score_indices_data)
{
  CUDA_KERNEL_LOOP(idx, nthreads) {
    // For each of the beam_size best expansions of this beam, 
    // calculate the score summed over the resulting partial sequence.
	  bool beam_complete = seq && seq[idx*timestep+timestep-1] == end_of_sequence;
    int c = 0; // candidate index
    int e = 0; // output index
    while (e < beam_size) { // expansion
      outer_loop:
      int index = output_score_indices_data[idx*vocab_size+c];
      int word = index % vocab_size;
      int offset = idx*beam_size + e;
      if (timestep == 0 && idx % beam_size != 0) {
        // All beams are the same in first iteration, so avoid duplicating multiple copies
        break;
      } else if (beam_complete) {
        // Keep, but don't expand completed beam
        score_indices_data[offset] = index;
        score_data[offset] = partial_score[idx];
        // Don't want multiple copies of completed beam
        break;
      } else if (prevent_repeats && seq && seq[idx*timestep+timestep-1] == Dtype(word)) {
        // Don't allow an immediate repeat of any word
        c++;
        continue;
      } else if (allowed_multiple_size > 0) {
        // Don't allow repeat of any word not in allowed_multiple
        for (int t=0; t<timestep; ++t) {
          Dtype prev_word = seq[idx*timestep+t];
          if (Dtype(word) == prev_word) {
            bool allowed_multiple = false;
            int r=0;
            for (; r<allowed_multiple_size; ++r) {
              if (Dtype(word) == allowed_multiple_data[r]) {
                allowed_multiple = true;
                break;
              }
            }
            if (!allowed_multiple) {
              c++;
              goto outer_loop;
            }
          }
        }
      }
      score_indices_data[offset] = index;
      score_data[offset] = output_score_sorted[idx*vocab_size+c];
      if (timestep > 0) { // Add score of existing partial sequence
        score_data[offset] += partial_score[idx];
      }
      e++;
      c++;
    }
  }
}


template <typename Dtype>
__global__ void ForwardOutput(
  const int nthreads,
  const int beam_size,
  const int vocab_size,
  const Dtype end_of_sequence,
  const int timestep,
  const Dtype* score_data,
  const int* score_indices_data,
  const Dtype* seq,
  int* source_map_data,
  Dtype* next_partial_sum,
  Dtype* seq_next,
  Dtype* next_input
  )
{
  CUDA_KERNEL_LOOP(idx, nthreads) {
    const int n = idx / beam_size; // which element in batch
    const int beam = idx % beam_size;
    const int index = score_indices_data[n*beam_size*beam_size + beam];
    const int src_idx = index / vocab_size;
    source_map_data[idx] = src_idx;
    const int word = index % vocab_size;
    for (int s = 0; s < timestep; ++s) {
      seq_next[idx*(timestep+1) + s] = seq[src_idx*timestep + s];
    }
    next_partial_sum[idx] = score_data[n*beam_size*beam_size + beam];
    bool beam_complete = seq && seq[src_idx*timestep+timestep-1] == end_of_sequence;
    if (beam_complete) {
      seq_next[idx*(timestep+1) + timestep] = end_of_sequence;
      next_input[idx] = end_of_sequence;
    } else {
      seq_next[idx*(timestep+1) + timestep] = word;
      next_input[idx] = word;
    }
  }
}


template <typename Dtype>
__global__ void RecurrentForward(
    const int nthreads,
    const int timestep,
    const int chunk_size,
    const Dtype end_of_sequence,
    const int* source_map_data,
    const Dtype* seq, 
    const Dtype* bottom_data,
    Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    for (int i=0; i<chunk_size; ++i){
      top_data[index*chunk_size+i] = Dtype(0);
    }
    for (int i=0; i<nthreads; ++i){
      if (source_map_data[i] == index){
        if (!seq || seq[index*timestep+timestep-1] != end_of_sequence){
          // Sequence has not ended
          for (int j=0; j<chunk_size; ++j){
            top_data[i*chunk_size+j] = bottom_data[index*chunk_size+j];
          }
        }
      }
    }
  }
}

template <typename Dtype>
void BeamSearchNodeLayer<Dtype>::recurrent_forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  int* source_map_data = source_map_.mutable_gpu_data();
	const Dtype* seq = NULL;
  if (has_input_scores_) {
    seq = bottom[1]->gpu_data();
  }
	for (int i = NonRecurrentOutputs(); i<top.size(); ++i){
		int bottom_ix = has_input_scores_ ? i : i - 2;
		// NOLINT_NEXT_LINE(whitespace/operators)
		RecurrentForward<Dtype> << <CAFFE_GET_BLOCKS(batch_size_), CAFFE_CUDA_NUM_THREADS >> >(
			batch_size_, timestep_, top[i]->shape(1), end_of_sequence_, source_map_data, seq,
			bottom[bottom_ix]->gpu_data(), top[i]->mutable_gpu_data());
	}
	CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
void BeamSearchNodeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // First, sort all possible beam expansions, so we can ignore all but the beam_size best
  if (has_input_scores_) {
    caffe_copy(bottom[2]->count(), bottom[2]->gpu_data(), exp_.mutable_gpu_data());    
  } else {
    caffe_copy(bottom[0]->count(), bottom[0]->gpu_data(), exp_.mutable_gpu_data());
  }
  thrust::device_ptr<int> index = thrust::device_pointer_cast(exp_indices_.mutable_gpu_data());
  thrust::sequence(index, index + exp_indices_.count()); 
  reverse_gpu_sort(exp_, exp_indices_, vocab_size_);
  
  // Second, sum existing partial scores with their best beam_size possible expansions
  const Dtype* allowed_multiple_data = NULL;
  if (allowed_multiple_size_ > 0) {
    allowed_multiple_data = allowed_multiple_.gpu_data();
  }
  const Dtype* partial_sum = NULL;
  const Dtype* seq = NULL;
  if (has_input_scores_) {
    partial_sum = bottom[0]->gpu_data();
    seq = bottom[1]->gpu_data();
  }
  const Dtype* output_score_sorted = exp_.gpu_data();
  const int* output_score_indices_data = exp_indices_.gpu_data();
  Dtype* score_data = score_.mutable_gpu_data();
  int* score_indices_data = score_indices_.mutable_gpu_data();
  caffe_gpu_set(score_.count(), Dtype(-FLT_MAX), score_data);
  SumScores<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
    <<<CAFFE_GET_BLOCKS(batch_size_), CAFFE_CUDA_NUM_THREADS>>>(
      batch_size_, timestep_, end_of_sequence_, beam_size_, vocab_size_, 
      prevent_repeats_, allowed_multiple_size_, allowed_multiple_data, seq, 
      output_score_sorted, output_score_indices_data,
      partial_sum, score_data, score_indices_data);
  CUDA_POST_KERNEL_CHECK;

  // Third, sort summed expansions
  reverse_gpu_sort(score_, score_indices_, beam_size_*beam_size_);
  
  // Fourth, generate outputs for next step, 
  // including recurrent connections and source_map_
  Dtype* next_partial_sum = top[0]->mutable_gpu_data();
  Dtype* next_seq = top[1]->mutable_gpu_data();
  Dtype* next_input = top[2]->mutable_gpu_data();
  int* source_map_data = source_map_.mutable_gpu_data();
  ForwardOutput<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)  
    << <CAFFE_GET_BLOCKS(batch_size_), CAFFE_CUDA_NUM_THREADS >> >(
    batch_size_, beam_size_, vocab_size_, end_of_sequence_, timestep_,
    score_.gpu_data(), score_indices_.gpu_data(), seq, source_map_data,
    next_partial_sum, next_seq, next_input);
  CUDA_POST_KERNEL_CHECK;
  recurrent_forward_gpu(bottom, top);
}


template <typename Dtype>
__global__ void ScoresBackward(
    const int nthreads,
    const int timestep,
    const int vocab_size,
    const Dtype end_of_sequence,
    const int* source_map_data,
    const Dtype* seq, 
    const Dtype* next_partial_sum_diff,
    const Dtype* next_input,
    Dtype* score_diff,
    Dtype* partial_sum_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    if (seq) {
      partial_sum_diff[index] = Dtype(0);
    }
    for (int i=0; i<vocab_size; ++i){
      score_diff[index*vocab_size+i] = Dtype(0);
    }
    for (int i=0; i<nthreads; ++i){    
      if (source_map_data[i] == index){
        Dtype diff = next_partial_sum_diff[i];
        // sum of scores and current score inputs
        if (seq) {
          partial_sum_diff[index] += diff;
        }
        if (!seq || seq[index*timestep+timestep-1] != end_of_sequence){
          // Sequence has not ended
          score_diff[index*vocab_size+int(next_input[i])] += diff;
        }
      }
    }
  }
}


template <typename Dtype>
__global__ void RecurrentBackward(
    const int nthreads,
    const int timestep,
    const int chunk_size,
    const Dtype end_of_sequence,
    const int* source_map_data,
    const Dtype* seq, 
    const Dtype* top_diff,
    Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    for (int i=0; i<chunk_size; ++i){
      bottom_diff[index*chunk_size+i] = Dtype(0);
    }
    for (int i=0; i<nthreads; ++i){    
      if (source_map_data[i] == index){
        if (!seq || seq[index*timestep+timestep-1] != end_of_sequence){
          // Sequence has not ended
          for (int j=0; j<chunk_size; ++j){
            bottom_diff[index*chunk_size+j] += top_diff[i*chunk_size+j];
          }
        }
      }
    }
  }
}

template <typename Dtype>
void BeamSearchNodeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  CHECK(!(has_input_scores_ && propagate_down[1])) 
        << "cannot backprop to partial sequence inputs.";
  const int* source_map_data = source_map_.gpu_data();
  const Dtype* next_partial_sum_diff = top[0]->gpu_diff();
  const Dtype* next_input = top[2]->gpu_data();
  Dtype* partial_sum_diff = NULL;
  Dtype* score_diff = NULL;
  const Dtype* seq = NULL;
  if (has_input_scores_) {
    partial_sum_diff = bottom[0]->mutable_gpu_diff();
    score_diff = bottom[2]->mutable_gpu_diff();
    seq = bottom[1]->gpu_data();
  } else {
    score_diff = bottom[0]->mutable_gpu_diff();
  }
  int threads = batch_size_;
  // NOLINT_NEXT_LINE(whitespace/operators)
  ScoresBackward<Dtype> <<<CAFFE_GET_BLOCKS(threads), CAFFE_CUDA_NUM_THREADS>>>(
      threads, timestep_, vocab_size_, end_of_sequence_, source_map_data, seq, 
      next_partial_sum_diff, next_input, score_diff, partial_sum_diff);
  CUDA_POST_KERNEL_CHECK;
  for(int i=NonRecurrentOutputs(); i<top.size(); ++i){
    int bottom_ix = has_input_scores_ ? i : i-2;
    // NOLINT_NEXT_LINE(whitespace/operators)
    RecurrentBackward<Dtype> <<<CAFFE_GET_BLOCKS(threads), CAFFE_CUDA_NUM_THREADS>>>(
        threads, timestep_, top[i]->shape(1), end_of_sequence_, source_map_data, seq, 
        top[i]->gpu_diff(), bottom[bottom_ix]->mutable_gpu_diff());
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(BeamSearchNodeLayer);

}  // namespace caffe
