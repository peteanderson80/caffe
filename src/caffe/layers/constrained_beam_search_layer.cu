#include <algorithm>
#include <cfloat>
#include <vector>
#include <iomanip>

#include "caffe/layers/constrained_beam_search_layer.hpp"

namespace caffe {



template <typename Dtype>
__global__ void UpdateConstrainedScore(
  const int nthreads,
  int timestep,
  int end_of_sequence,
  int beam_size,
  int vocab_size,
  int sequence_length,
  int num_states,
  bool prevent_repeats,
  int allowed_multiple_size,
  const Dtype* allowed_multiple_data,
  const Dtype* sequence_output_prev,
  const Dtype* input_sequence_data,
  const int num_conjunctions,
  const int num_disjunctions,
  const Dtype* constraint_data,
  const Dtype* output_score_sorted,
  const Dtype* output_score_data,
  const int* output_score_indices_data,
  const Dtype* partial_score,
  Dtype* score_data,
  int* score_indices_data)
{
  CUDA_KERNEL_LOOP(idx, nthreads) {
    // One kernel for each partial sequence. Consider the best possible expansions, 
    // and calculate the score for the resulting sequence. Save the results in score_data
    // with the matching indices in score_indices_data. Note: score_data starts as -FLT_MAX.
    const bool beam_complete = (timestep > 0) && (input_sequence_data[idx] == Dtype(end_of_sequence));
    const unsigned int state = (idx / beam_size) % num_states;
    const bool is_constrained = (state > 0);
    unsigned int num_constraints; // determined by the total bits set in state
    unsigned int state_copy = state;
    for (num_constraints = 0; state_copy; state_copy >>= 1){
      num_constraints += state_copy & 1;
    }
    // At most we need to consider this many expansions for this beam.
    int max_expansions = beam_size + num_conjunctions * num_disjunctions;

    // Expand own partial sequence
    int c = 0; // candidate index
    int e = 0; // output index
    while (e < beam_size) { // expansion
      outer_loop:
      int index = output_score_indices_data[idx*vocab_size+c];
      int word = index % vocab_size;
      int offset = idx*max_expansions + e;
      score_indices_data[offset] = index;
      if (timestep == 0 && (idx % beam_size != 0)) {
        // All beams are the same in first iteration (empty), so avoid duplicates
        // by not expanding unless this is beam 0
        break;
      } else if (timestep < num_constraints) {
        // Can't self-expand constrained beams until they are initially populated
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
      score_data[offset] = output_score_sorted[idx*vocab_size+c]; // score for next word
      if (timestep > 0) { // Add score of existing partial sequence
        score_data[offset] += partial_score[idx];
      }
      e++;
      c++;
    }
    // Expand partial sequences from other states
    if (is_constrained && timestep >= (num_constraints-1)) {
      // Add further beam expansions by adding constraint words to the unconstrained beams
      const int n = idx / (num_states*beam_size); // batch index
      for (int c=0; c<num_conjunctions; ++c) {
        // Check if this state is conditional on these alternatives, by checking bit in position c
        if (state & (1<<c)) {
          for (int a=0; a<num_disjunctions; ++a) {
            int constraint_word = constraint_data[(n*num_conjunctions+c)*num_disjunctions+a];
            if (constraint_word == end_of_sequence) {
              continue; // ignore
            }
            unsigned int src_state = state & ~(1<<c); // Src state is same but with zeroed bit c
            int offset = idx*max_expansions + beam_size + c*num_disjunctions + a;
            int src_idx = idx-(state-src_state)*beam_size;
            const bool src_beam_complete = (input_sequence_data[src_idx] == Dtype(end_of_sequence));
            if (src_beam_complete) {
              continue; // Can't add constraint words to a completed sequence
            }
            int index = src_idx * vocab_size + constraint_word; // Source state index
            Dtype next_word_score = output_score_data[index];
            if (timestep==0 && num_constraints==1) {
              // If this state conditional on only one word, insert conditional word at t=0...
              if (idx % beam_size == 0) {
                // ...but only for one beam
                score_data[offset] = next_word_score;
                score_indices_data[offset] = index;
              }
            } else if (input_sequence_data[src_idx] != Dtype(end_of_sequence)) { 
              // Sequence not complete, so further expand
              score_indices_data[offset] = index;
              if (prevent_repeats && input_sequence_data[src_idx] == constraint_word) {
                score_data[offset] = Dtype(-FLT_MAX);
              } else if (allowed_multiple_size > 0) {
                // Don't allow repeat of any word not in allowed_multiple
                bool reject = false;
                for (int t=0; t<timestep; ++t) {
                  Dtype prev_word = sequence_output_prev[src_idx*sequence_length+t];
                  if (Dtype(constraint_word) == prev_word) {
                    bool allowed_multiple = false;
                    int r=0;
                    for (; r<allowed_multiple_size; ++r) {
                      if (Dtype(constraint_word) == allowed_multiple_data[r]) {
                        allowed_multiple = true;
                        break;
                      }
                    }
                    if (!allowed_multiple) {
                      reject = true;
                    }
                  }
                }
                if (reject) {
                  score_data[offset] = Dtype(-FLT_MAX);
                } else {
                  score_data[offset] = next_word_score + partial_score[src_idx];
                }
              } else {
                score_data[offset] = next_word_score + partial_score[src_idx];
              }
            }
          }
        }
      }
    }
  }
}

template <typename Dtype>
void ConstrainedBeamSearchLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
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
    const Dtype* allowed_multiple_data = NULL;
    if (this->allowed_multiple_size_ > 0) {
      allowed_multiple_data = this->allowed_multiple_.gpu_data();
    }
    const Dtype* sequence_output_prev = top[0]->gpu_data();
    const Dtype* constraint_data = bottom[0]->gpu_data();
    const int num_conjunctions = bottom[0]->shape(1);
    const int num_disjunctions = bottom[0]->shape(2);
    const Dtype* output_score_sorted = this->output_score_->gpu_diff();
    const Dtype* output_score_data = this->output_score_->gpu_data();
    const int* output_score_indices_data = this->output_score_indices_.gpu_data();
    Dtype* score_data = this->score_.mutable_gpu_data();
    int* score_indices_data = this->score_indices_.mutable_gpu_data();
    caffe_gpu_set(this->score_.count(), Dtype(-FLT_MAX), score_data);
    UpdateConstrainedScore<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, timestep, this->end_of_sequence_, this->beam_size_, this->vocab_size_, this->sequence_length_, 
        this->num_states_, this->prevent_repeats_, this->allowed_multiple_size_, allowed_multiple_data, 
        sequence_output_prev, input_sequence_data, num_conjunctions, num_disjunctions, 
        constraint_data, output_score_sorted, output_score_data, output_score_indices_data, partial_score, 
        score_data, score_indices_data);
    CUDA_POST_KERNEL_CHECK;

    /*
    cudaDeviceSynchronize();
    constraint_data = bottom[0]->cpu_data();
    std::ostringstream os1;
    os1 << "First batch item score pre-sorting at t=" << timestep << ", conditioned on ";
    for (int l=0; l<num_conjunctions*num_disjunctions; ++l) {
      os1 << constraint_data[l] << ", ";
    }
    LOG(INFO) << os1.str();
    score_data = this->score_.mutable_cpu_data();
    for (int l=0; l<this->beam_size_*this->num_states_; ++l) {
      std::ostringstream os;
      for (int k=0; k<(this->beam_size_+num_conjunctions*num_disjunctions); ++k) {
        os << std::setfill(' ') << std::setw(4) << score_data[l*(this->beam_size_+num_conjunctions*num_disjunctions)+k] << " ";
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
    for (int l=0; l<num_conjunctions*num_disjunctions; ++l) {
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

INSTANTIATE_LAYER_GPU_FORWARD(ConstrainedBeamSearchLayer);

}  // namespace caffe
