#include <algorithm>
#include <cfloat>
#include <vector>
#include <iomanip>

#include "caffe/layers/sequence_constrained_beam_search_layer.hpp"

namespace caffe {



template <typename Dtype>
__global__ void UpdateConstrainedSequenceScore(
  const int nthreads,
  int timestep,
  int end_of_sequence,
  int beam_size,
  int vocab_size,
  int num_states,
  int sequence_length,
  bool prevent_repeats,
  int allowed_multiple_size,
  const Dtype* allowed_multiple_data,
  const Dtype* sequence_output_prev,
  const Dtype* input_sequence_data,
  int input_sequence_length,
  const Dtype* constraint_data,
  const Dtype* output_score_sorted,
  const Dtype* output_score_data,
  const int* output_score_indices_data,
  const Dtype* partial_score,
  Dtype* score_data,
  int* score_indices_data)
{
  CUDA_KERNEL_LOOP(idx, nthreads) {
    // For each beam expansion, calculate the score summed
    // over the resulting partial sequence.
    const bool beam_complete = (timestep > 0) && (input_sequence_data[idx] == Dtype(end_of_sequence));
    const unsigned int state = (idx / beam_size) % num_states;
    int max_expansions = beam_size + 1;
    const int n = idx / (num_states*beam_size); // batch index
    int last_state = 0;
    for(int i=0; i<input_sequence_length; ++i) {
      if (constraint_data[n*input_sequence_length+i] == end_of_sequence) {
        break;
      }
      last_state += 1;
    }
    // Expand own beams - but only if first or last state
    if (state == 0 || state == last_state) {
      int c = 0; // candidate index
      int e = 0; // output index
      while (e < beam_size) {
        outer_loop:
        int index = output_score_indices_data[idx*vocab_size+c];
        int word = index % vocab_size;
        int offset = idx*max_expansions + e;
        if (timestep == 0 && (idx % beam_size != 0)) {
          // All beams are the same in first iteration (empty), so avoid duplicates
          // by not expanding unless this is beam 0
          e = beam_size;
          continue;
        } else if (timestep < state) {
          // Can't self-expand constrained beams until they are initially populated
          e = beam_size;
          continue;
        } else if (beam_complete) {
          // Keep, but don't expand completed beam
          score_indices_data[offset] = index;
          score_data[offset] = partial_score[idx];
          // Don't want multiple copies of completed beam
          e = beam_size;
          continue;
        } else if (prevent_repeats && input_sequence_data[idx] == Dtype(word)) {
          // Don't prevent an immediate repeat of any word
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
        score_indices_data[offset] = index;
        score_data[offset] = output_score_sorted[idx*vocab_size+c];
        if (timestep > 0) { // Add score of existing partial sequence
          score_data[offset] += partial_score[idx];
        }
        e++;
        c++;
      }
    }
    // Expand beams from the previous state
    if ((state > 0) && (timestep >= (state-1))) {
      // Add further beam expansions by adding constraint words to the unconstrained beams
      int constraint_word = constraint_data[n*input_sequence_length+state-1];
      if (constraint_word != end_of_sequence) { // otherwise ignore
        int offset = idx*max_expansions + beam_size;
        int src_idx = idx-beam_size; // same beam in previous state
        int index = src_idx * vocab_size + constraint_word; // Source state index
        Dtype incremental_score = output_score_data[index];
        if (timestep==0 && state==1) {
          // If this state is constrained on only one word, insert constraint word at t=0...
          if ((idx % beam_size) == 0) {
            // ...but only for one beam
            score_data[offset] = incremental_score;
            score_indices_data[offset] = index;
          }
        } else if (input_sequence_data[src_idx] != Dtype(end_of_sequence)) { 
          // Sequence not complete, so further expand
          score_indices_data[offset] = index;
          score_data[offset] = incremental_score + partial_score[src_idx];
        }
      }
    }
  }
}

template <typename Dtype>
void SequenceConstrainedBeamSearchLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const int batch_size = bottom[0]->shape(0);

  // Copy bottom input into the beam search net, duplicating for size of beam search
  this->copy_bottom_inputs_gpu(bottom);

  // Zero sequence, input and recurrent connections for first pass
  this->clear_recurrent_inputs_gpu();

  for (int timestep = 0; timestep < this->sequence_length_; ++timestep) {

    this->net_->Forward();
    this->sort_beam_expansions_gpu();

    // Replaces call to sum_expansion_scores_gpu
    const int nthreads = this->input_sequence_->shape(0);
    const Dtype* partial_score = top[1]->gpu_data();
    const Dtype* input_sequence_data = this->input_sequence_->gpu_data();
    const Dtype* constraint_data = bottom[0]->gpu_data();
    const Dtype* allowed_multiple_data = NULL;
    if (this->allowed_multiple_size_ > 0) {
      allowed_multiple_data = this->allowed_multiple_.gpu_data();
    }
    const Dtype* sequence_output_prev = top[0]->gpu_data();
    const int input_sequence_length = bottom[0]->shape(-1);
    const Dtype* output_score_sorted = this->output_score_->gpu_diff();
    const Dtype* output_score_data = this->output_score_->gpu_data();
    const int* output_score_indices_data = this->output_score_indices_.gpu_data();
    Dtype* score_data = this->score_.mutable_gpu_data();
    int* score_indices_data = this->score_indices_.mutable_gpu_data();
    caffe_gpu_set(this->score_.count(), Dtype(-FLT_MAX), score_data);
    UpdateConstrainedSequenceScore<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, timestep, this->end_of_sequence_, this->beam_size_, this->vocab_size_, this->num_states_, 
        this->sequence_length_, this->prevent_repeats_, this->allowed_multiple_size_, allowed_multiple_data, 
        sequence_output_prev, input_sequence_data, input_sequence_length, constraint_data, output_score_sorted, 
        output_score_data, output_score_indices_data, partial_score, score_data, score_indices_data);
    CUDA_POST_KERNEL_CHECK;

    /*
    cudaDeviceSynchronize();
    constraint_data = bottom[0]->cpu_data();
    std::ostringstream os1;
    os1 << "First batch item score pre-sorting at t=" << timestep << ", conditioned on ";
    for (int l=0; l<bottom[0]->shape(1); ++l) {
      os1 << constraint_data[l] << ", ";
    }
    LOG(INFO) << os1.str();
    score_data = this->score_.mutable_cpu_data();
    for (int l=0; l<this->beam_size_*this->num_states_; ++l) {
      std::ostringstream os;
      for (int k=0; k<(this->beam_size_+1); ++k) {
        os << std::setfill(' ') << std::setw(4) << score_data[l*(this->beam_size_+1)+k] << " ";
      }
      LOG(INFO) << os.str();
      if ((l+1) % this->beam_size_ == 0) {
        LOG(INFO) << " -------------------------- ";
      }
    }*/

    // Find the overall beam_size best sequences for each input
    this->sort_scores_gpu();

    // Save outputs
    this->generate_output(top, timestep);

    /*
    cudaDeviceSynchronize();
    constraint_data = bottom[0]->cpu_data();
    std::ostringstream os;
    os << "First batch item beams at t=" << timestep << ", conditioned on ";
    for (int l=0; l<bottom[0]->shape(1); ++l) {
      os << constraint_data[l] << ", ";
    }
    LOG(INFO) << os.str();
    const Dtype* sequence_output = top[0]->cpu_data();
    const Dtype* score_output = top[1]->cpu_data();
    for (int l=0; l<this->beam_size_*this->num_states_; ++l) {
      std::ostringstream os;
      for (int k=0; k<this->sequence_length_; ++k) {
        os << std::setfill(' ') << std::setw(4) << sequence_output[l*this->sequence_length_+k] << " ";
      }
      os << std::setfill(' ') << std::setw(4) << score_output[l];
      LOG(INFO) << os.str();
      if ((l+1) % this->beam_size_ == 0) {
        LOG(INFO) << " -------------------------- ";
      }
    }*/

    bool exiting = (timestep == this->sequence_length_-1);
    if (!exiting) {
      //Check for early exit
      cudaDeviceSynchronize();
      exiting = true;
      const Dtype* sequence_output = top[0]->cpu_data();
      for (int idx = 0; idx < nthreads; ++idx) {
        if (sequence_output[idx*this->sequence_length_+timestep] != Dtype(this->end_of_sequence_)){
          exiting = false;
        }
      }
    }
    if (exiting){
      break;
    }
    this->copy_back_recurrent_inputs_gpu(timestep);
  }
}

INSTANTIATE_LAYER_GPU_FORWARD(SequenceConstrainedBeamSearchLayer);

}  // namespace caffe
