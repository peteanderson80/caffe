#include <algorithm>
#include <cfloat>
#include <vector>
#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>

#include "caffe/layers/base_beam_search_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BaseBeamSearchLayer<Dtype>::copy_bottom_inputs_gpu(const vector<Blob<Dtype>*>& bottom) {
  // Copy bottom input into the beam search net, duplicating for size of beam search
  const int batch_size = bottom[0]->shape(0);
  for (int i = 0; i < bottom_connections_.size(); ++i) {
    int bottom_ix = bottom_connections_[i].first;
    const Dtype* src_data = bottom[bottom_ix]->cpu_data();
    Dtype* tgt_data = bottom_connections_[i].second->mutable_cpu_data();
    for (int k = 0; k < batch_size; ++k) {
      for (int j = 0; j < this->beam_size_ * this->num_states_; ++j) { // Multiple copies to accomodate beam size
        caffe_copy(bottom[bottom_ix]->count()/batch_size, src_data, tgt_data);
        tgt_data += bottom[bottom_ix]->count()/batch_size;
      }
      src_data += bottom[bottom_ix]->count()/batch_size;
    }
  }
}

template <typename Dtype>
void BaseBeamSearchLayer<Dtype>::clear_recurrent_inputs_gpu() {
  caffe_gpu_set(input_sequence_->count(), Dtype(end_of_sequence_), input_sequence_->mutable_gpu_data());
  for (int i = 0; i < recurrent_connections_.size(); ++i) {
    caffe_gpu_set(recurrent_connections_[i].second->count(), Dtype(0),
              recurrent_connections_[i].second->mutable_gpu_data());
  }
}

template <typename Dtype>
__global__ void RecycleState(
  const int nthreads,
  int beam_size,
  int end_of_sequence,
  const unsigned int timestep,
  const int vocab_size,
  const int* index_data,
  const int index_data_chunk_size,
  Dtype* input_sequence_data_prev,
  Dtype* input_sequence_data,
  int num_recurrent_connections,
  const Dtype** from,
  Dtype** to,
  int* count)
{
  CUDA_KERNEL_LOOP(idx, nthreads) {
    const int n = idx / beam_size; // which element in batch
    const int beam = idx % beam_size;
    const int index = index_data[n*index_data_chunk_size+beam];
    const int src_idx = index / vocab_size;
    const int word_index = index % vocab_size;
    if (timestep > 0 && input_sequence_data_prev[src_idx] == end_of_sequence) {
      input_sequence_data[idx] = end_of_sequence;
    } else {
      input_sequence_data[idx] = word_index;
    }
    for (int i=0; i<num_recurrent_connections; ++i) {
      int chunk_size = count[i]/nthreads;
      const Dtype* src = from[i] + src_idx*chunk_size;
      Dtype* dst = to[i] + idx*chunk_size;
      for (int j=0; j<chunk_size; ++j) {
        dst[j] = src[j];
      }
    }
  }
}

template <typename Dtype>
void BaseBeamSearchLayer<Dtype>::copy_back_recurrent_inputs_gpu(const unsigned int timestep) {
  // Setup next input sequence and recurrent blob inputs
  Dtype* input_sequence_data = this->input_sequence_->mutable_gpu_data();
  // temp copy of previous input
  Dtype* input_sequence_data_prev = this->input_sequence_->mutable_gpu_diff();
  caffe_copy(this->input_sequence_->count(), input_sequence_data, 
      input_sequence_data_prev);

  const int* index_data = score_indices_.gpu_data();
  const int index_data_chunk_size = score_.count()/score_.shape(0);
  thrust::device_vector<const Dtype*> from;
  thrust::device_vector<Dtype*> to;
  thrust::device_vector<int> counts;
  for (int i = 0; i < this->recurrent_connections_.size(); ++i) {
    from.push_back(this->recurrent_connections_[i].first->gpu_data());
    to.push_back(this->recurrent_connections_[i].second->mutable_gpu_data());
    counts.push_back(this->recurrent_connections_[i].first->count());
  }
  const Dtype** f = thrust::raw_pointer_cast(&from[0]);
  Dtype** t = thrust::raw_pointer_cast(&to[0]);
  int* c = thrust::raw_pointer_cast(&counts[0]);
  const int nthreads = input_sequence_->shape(0);
  RecycleState<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
  <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
    nthreads, this->beam_size_, end_of_sequence_, timestep, this->vocab_size_, index_data,
    index_data_chunk_size, input_sequence_data_prev, input_sequence_data, counts.size(), f, t, c);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void BaseBeamSearchLayer<Dtype>::sort_beam_expansions_gpu() {
  // Sorted scores will be held in output_score_ diff.
  Dtype* output_score_data = output_score_->mutable_gpu_diff();
  caffe_copy(output_score_->count(), output_score_->gpu_data(), output_score_data);
  int* index_data = output_score_indices_.mutable_gpu_data();
  int* idx_index_data = output_score_indices_.mutable_gpu_diff(); // Temp usage of diff
  thrust::device_ptr<Dtype> output_score = thrust::device_pointer_cast(output_score_data);
  thrust::device_ptr<int> index = thrust::device_pointer_cast(index_data);
  thrust::device_ptr<int> idx_index = thrust::device_pointer_cast(idx_index_data);
  const int score_count = output_score_->count();

  // Initialise idx_index to idx in blocks of vocab_size
  thrust::transform(thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(score_count),
                    thrust::make_constant_iterator(vocab_size_),
                    idx_index,
                    thrust::divides<int>() );

  // Initialise index to a global index offset
  thrust::sequence(index, index + score_count);

  // Back-to-back sorting is a fast way of concurrently sorting multiple arrays
  // Use of the 'greater' sorting function will reverse sort (highest scores first)
  thrust::stable_sort_by_key(output_score, output_score + score_count,
                              thrust::make_zip_iterator(thrust::make_tuple(idx_index, index)),
                              thrust::greater<Dtype>()
                            );
  thrust::stable_sort_by_key(idx_index, idx_index + score_count,
                              thrust::make_zip_iterator(thrust::make_tuple(output_score, index)));
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void UpdateScore(
  const int nthreads,
  int timestep,
  int end_of_sequence,
  int beam_size,
  int vocab_size,
  int sequence_length,
  bool prevent_repeats,
  int allowed_multiple_size,
  const Dtype* allowed_multiple_data,
  const Dtype* sequence_output_prev,
  const Dtype* input_sequence_data,
  const Dtype* output_score_sorted,
  const int* output_score_indices_data,
  const Dtype* partial_score,
  Dtype* score_data,
  int* score_indices_data)
{
  CUDA_KERNEL_LOOP(idx, nthreads) {
    // One kernel for each partial sequence. Consider the top beam_size possible expansions, 
    // and calculate the score for the resulting sequence. Note: score_data starts as -FLT_MAX.
    const bool beam_complete = (timestep > 0) && (input_sequence_data[idx] == Dtype(end_of_sequence));
    int c = 0; // candidate index
    int e = 0; // output index
    while (e < beam_size) { // expansion
      outer_loop:
      int index = output_score_indices_data[idx*vocab_size+c];
      int word = index % vocab_size;
      int offset = idx*beam_size + e;
      score_indices_data[offset] = index;
      if (timestep == 0 && (idx % beam_size != 0)) {
        // All beams are the same in first iteration (empty), so avoid duplicates
        // by not expanding unless this is beam 0
        break;
      } else if (beam_complete) {
        // Keep, but don't expand completed beam
        score_data[offset] = partial_score[idx];
        // Don't want multiple copies of completed beam
        break;
      } else if (prevent_repeats && input_sequence_data[idx] == Dtype(word)) {
        // Don't allow an immediate repeat of any word
        c++;
        continue;
      } else if (allowed_multiple_size > 0) {
        // Don't allow repeat of any word not in allowed_multiple
        for (int t=0; t<timestep; ++t) {
          Dtype prev_word = sequence_output_prev[idx*sequence_length+t];
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
void BaseBeamSearchLayer<Dtype>::sum_expansion_scores_gpu(const vector<Blob<Dtype>*>& top,
        const unsigned int timestep) {
  const int nthreads = input_sequence_->shape(0);
  const Dtype* input_sequence_data = input_sequence_->gpu_data();
  const Dtype* allowed_multiple_data = NULL;
  if (allowed_multiple_size_ > 0) {
    allowed_multiple_data = this->allowed_multiple_.gpu_data();
  }
  const Dtype* sequence_output_prev = top[0]->gpu_data();
  const Dtype* partial_score = top[1]->gpu_data();
  const Dtype* output_score_sorted = output_score_->gpu_diff();
  const int* output_score_indices_data = output_score_indices_.gpu_data();
  Dtype* score_data = score_.mutable_gpu_data();
  int* score_indices_data = score_indices_.mutable_gpu_data();
  caffe_gpu_set(this->score_.count(), Dtype(-FLT_MAX), score_data);
  UpdateScore<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
    <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
      nthreads, timestep, end_of_sequence_, beam_size_, vocab_size_, sequence_length_,
      prevent_repeats_, allowed_multiple_size_, allowed_multiple_data, 
      sequence_output_prev, input_sequence_data, output_score_sorted, output_score_indices_data,
      partial_score, score_data, score_indices_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void BaseBeamSearchLayer<Dtype>::sort_scores_gpu() {
  Dtype* score_data = score_.mutable_gpu_data();
  int* index_data = score_indices_.mutable_gpu_data();
  int* idx_index_data = score_indices_.mutable_gpu_diff(); // Temp usage of diff
  thrust::device_ptr<Dtype> score = thrust::device_pointer_cast(score_data);
  thrust::device_ptr<int> index = thrust::device_pointer_cast(index_data);
  thrust::device_ptr<int> idx_index = thrust::device_pointer_cast(idx_index_data);

  // Now, initialize to idx in blocks
  thrust::transform(thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(score_.count()),
                    thrust::make_constant_iterator(score_.count()/score_.shape(0)),
                    idx_index,
                    thrust::divides<int>() );
  thrust::stable_sort_by_key(score, score + score_.count(),
                              thrust::make_zip_iterator(thrust::make_tuple(idx_index, index)),
                              thrust::greater<Dtype>() // greatest score first
                            );
  thrust::stable_sort_by_key(idx_index, idx_index + score_.count(),
                              thrust::make_zip_iterator(thrust::make_tuple(score, index)));
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void GenerateOutput(
  const int nthreads,
  int beam_size,
  const int vocab_size,
  int num_states,
  int end_of_sequence,
  int timestep,
  int sequence_length,
  int max_expansions,
  const Dtype* score_data,
  const int* score_indices_data,
  const Dtype* sequence_output_prev,
  Dtype* sequence_output,
  const Dtype* score_sequence_output_prev,
  Dtype* score_sequence_output,
  Dtype* score_output)
{
  CUDA_KERNEL_LOOP(idx, nthreads) {
    const int n = idx / (num_states*beam_size); // which element in batch
    const int beam = idx % beam_size;
    const int state = (idx / beam_size) % num_states;
    const int index = score_indices_data[(n*num_states + state)*beam_size*max_expansions + beam];
    const int idx_index = index / vocab_size;
    const int word_index = index % vocab_size;
    score_output[idx] = score_data[(n*num_states + state)*beam_size*max_expansions + beam];
    Dtype partial_score_sum = 0;
    for (int s = 0; s < timestep; ++s) {
      sequence_output[idx*sequence_length + s] = sequence_output_prev[idx_index*sequence_length + s];
      score_sequence_output[idx*sequence_length + s] = score_sequence_output_prev[idx_index*sequence_length + s];
      partial_score_sum += score_sequence_output_prev[idx_index*sequence_length + s];
    }
    bool beam_complete = (timestep > 0) &&
        (sequence_output_prev[idx_index*sequence_length+timestep-1] == Dtype(end_of_sequence));
    if (beam_complete) {
      sequence_output[idx*sequence_length + timestep] = Dtype(end_of_sequence);
      score_sequence_output[idx*sequence_length + timestep] = Dtype(0);
    } else {
      sequence_output[idx*sequence_length + timestep] = word_index;
      score_sequence_output[idx*sequence_length + timestep] = score_output[idx] - partial_score_sum;
    }
  }
}

template <typename Dtype>
void BaseBeamSearchLayer<Dtype>::generate_output(const vector<Blob<Dtype>*>& top, 
      const unsigned int timestep) {
  Dtype* score_output = top[1]->mutable_gpu_data();
  Dtype* sequence_output = top[0]->mutable_gpu_data();
  Dtype* sequence_output_prev = top[0]->mutable_gpu_diff();
  caffe_copy(top[0]->count(), sequence_output, sequence_output_prev);
  Dtype* score_sequence_output = top[2]->mutable_gpu_data();
  Dtype* score_sequence_output_prev = top[2]->mutable_gpu_diff();
  caffe_copy(top[2]->count(), score_sequence_output, score_sequence_output_prev);
  const Dtype* score_data = this->score_.gpu_data();
  const int* score_indices_data = this->score_indices_.gpu_data();
  int max_expansions = score_.shape(-1);
  const int nthreads = input_sequence_->shape(0);
  GenerateOutput<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
  <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
    nthreads, this->beam_size_, this->vocab_size_, this->num_states_, this->end_of_sequence_, timestep,
    this->sequence_length_, max_expansions, score_data, score_indices_data, sequence_output_prev, sequence_output,
    score_sequence_output_prev, score_sequence_output, score_output);
  CUDA_POST_KERNEL_CHECK;
}

template void BaseBeamSearchLayer<float>::copy_bottom_inputs_gpu(const vector<Blob<float>*>& bottom);
template void BaseBeamSearchLayer<double>::copy_bottom_inputs_gpu(const vector<Blob<double>*>& bottom);
template void BaseBeamSearchLayer<float>::clear_recurrent_inputs_gpu();
template void BaseBeamSearchLayer<double>::clear_recurrent_inputs_gpu();
template void BaseBeamSearchLayer<float>::copy_back_recurrent_inputs_gpu(const unsigned int timestep);
template void BaseBeamSearchLayer<double>::copy_back_recurrent_inputs_gpu(const unsigned int timestep);
template void BaseBeamSearchLayer<float>::sort_beam_expansions_gpu();
template void BaseBeamSearchLayer<double>::sort_beam_expansions_gpu();
template void BaseBeamSearchLayer<float>::sum_expansion_scores_gpu(const vector<Blob<float>*>& top, unsigned int);
template void BaseBeamSearchLayer<double>::sum_expansion_scores_gpu(const vector<Blob<double>*>& top, unsigned int);
template void BaseBeamSearchLayer<float>::sort_scores_gpu();
template void BaseBeamSearchLayer<double>::sort_scores_gpu();
template void BaseBeamSearchLayer<float>::generate_output(const vector<Blob<float>*>& top, const unsigned int timestep);
template void BaseBeamSearchLayer<double>::generate_output(const vector<Blob<double>*>& top, const unsigned int timestep);

}  // namespace caffe
