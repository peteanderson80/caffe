#include <algorithm>
#include <vector>
#include <boost/concept_check.hpp>

#include "caffe/layers/beam_search_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"

namespace caffe {


template <typename Dtype>
void BeamSearchLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = bottom[0]->shape(0);

  // Copy bottom input into the beam search net, duplicating for size of beam search
  this->copy_bottom_inputs_cpu(bottom);

  // Zero sequence, input and recurrent connections for first pass
  this->clear_recurrent_inputs_cpu();

  std::vector<BeamExpansion> beams;
  beams.reserve(this->beam_size_ * this->beam_size_);

  for (int it = 0; it < this->sequence_length_; ++it) {

    bool early_exit = true;
    this->net_->Forward();

    Dtype* input_data = this->input_sequence_->mutable_cpu_data();
    const Dtype* output_data = this->output_score_->cpu_data();
    Dtype* sequence_data = top[0]->mutable_cpu_data();
    Dtype* score_data = top[1]->mutable_cpu_data();
    Dtype* score_seq_data = top[2]->mutable_cpu_data();

    for (int n = 0; n < batch_size; ++n) {
      beams.clear();
      // Expand all partial sequences in the current beam
      for (int b = 0; b < this->beam_size_; ++b) {
        BeamExpansion base_beam = {};
        base_beam.index = (n*this->beam_size_+b);
        base_beam.score = (it == 0) ? Dtype(0) : score_data[base_beam.index];
        for (int s=0; s<it; ++s) {
          int offset = base_beam.index*this->sequence_length_ + s;
          base_beam.words.push_back(sequence_data[offset]);
          base_beam.score_seq.push_back(score_seq_data[offset]);
        }
        if (it > 0 && sequence_data[base_beam.index*this->sequence_length_+it-1] == this->end_of_sequence_) {
          beams.push_back(base_beam); // Retain, but don't expand, a completed beam
        } else {
          early_exit = false;
          for (int w = 0; w < this->vocab_size_; ++w) {
            BeamExpansion ex(base_beam);
            Dtype score = output_data[base_beam.index*this->vocab_size_ + w];
            if (!this->prevent_repeats_ || ex.words.empty() || ex.words.back() != w) {
              ex.score += score;
              ex.score_seq.push_back(score);
              ex.words.push_back(w);
              beams.push_back(ex);
            }
          }
        }
        if (it == 0) {
          break; // All beams are the same in first iteration, avoid duplicates
        }
      }
      // Choose best beam_size partial sequences to store for next iteration
      std::partial_sort(beams.begin(), beams.begin()+this->beam_size_, beams.end());
      for (int b = 0; b < this->beam_size_; ++b) {
        BeamExpansion beam = beams[b];
        score_data[n*this->beam_size_+b] = beam.score;
        for (int s = 0; s < beam.words.size(); ++s) {
          sequence_data[(n*this->beam_size_+b)*this->sequence_length_ + s] = beam.words[s];
          score_seq_data[(n*this->beam_size_+b)*this->sequence_length_ + s] = beam.score_seq[s];
        }
        if (it < this->sequence_length_ -1) {
          // Setup inputs for next iteration
          input_data[n*this->beam_size_+b] = beam.words.back();
          for (int i = 0; i < this->recurrent_connections_.size(); ++i) {
            int chunk_size = this->recurrent_connections_[i].first->count()/(this->beam_size_*batch_size);
            caffe_copy(chunk_size, this->recurrent_connections_[i].first->cpu_data() + beam.index*chunk_size,
                      this->recurrent_connections_[i].second->mutable_cpu_data() + (n*this->beam_size_+b)*chunk_size);
          }
        }
      }
      /*
      if (n==0) {
        LOG(INFO) << "First item beams at t=" << it;
        sequence_data = top[0]->mutable_cpu_data();
        score_data = top[1]->mutable_cpu_data();
        for (int l=0; l<this->beam_size_; ++l) {
          std::ostringstream os;
          for (int k=0; k<this->sequence_length_; ++k) {
            os << std::setfill(' ') << std::setw(4) << sequence_data[l*this->sequence_length_+k] << " ";
          }
          os << std::setfill(' ') << std::setw(4) << score_data[l];
          LOG(INFO) << os.str();
        }
      }
      */
    }
    if (early_exit) {
      break;
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU_FORWARD(BeamSearchLayer, Forward);
#endif

INSTANTIATE_CLASS(BeamSearchLayer);
REGISTER_LAYER_CLASS(BeamSearch);

}  // namespace caffe
