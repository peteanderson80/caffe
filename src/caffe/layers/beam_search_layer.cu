#include <algorithm>
#include <vector>
#include <iomanip>

#include "caffe/layers/beam_search_layer.hpp"

namespace caffe {



template <typename Dtype>
void BeamSearchLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const int batch_size = bottom[0]->shape(0);
  const int nthreads = this->input_sequence_->shape(0);

  // Copy bottom input into the beam search net, duplicating for size of beam search
  this->copy_bottom_inputs_gpu(bottom);

  // Zero sequence, input and recurrent connections for first pass
  this->clear_recurrent_inputs_gpu();

  for (int timestep = 0; timestep < this->sequence_length_; ++timestep) {

    this->net_->Forward();
    this->sort_beam_expansions_gpu();

    this->sum_expansion_scores_gpu(top, timestep);

    // Find the overall beam_size best sequences for each input
    this->sort_scores_gpu();

    // Save outputs
    this->generate_output(top, timestep);

    /*
    cudaDeviceSynchronize();
    LOG(INFO) << "First item beams at t=" << timestep;
    const Dtype* sequence_output = top[0]->cpu_data();
    const Dtype* score_output = top[1]->cpu_data();
    for (int l=0; l<this->beam_size_; ++l) {
      std::ostringstream os;
      for (int k=0; k<this->sequence_length_; ++k) {
        os << std::setfill(' ') << std::setw(4) << sequence_output[l*this->sequence_length_+k] << " ";
      }
      os << std::setfill(' ') << std::setw(4) << score_output[l];
      LOG(INFO) << os.str();
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

INSTANTIATE_LAYER_GPU_FORWARD(BeamSearchLayer);

}  // namespace caffe
