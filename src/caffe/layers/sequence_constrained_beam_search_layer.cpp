#include <cmath>

#include "caffe/layers/sequence_constrained_beam_search_layer.hpp"


namespace caffe {


template <typename Dtype>
void SequenceConstrainedBeamSearchLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  BaseBeamSearchLayer<Dtype>::LayerSetUp(bottom, top);
  // Check the constraint inputs are ok
  CHECK_EQ(bottom[0]->num_axes(), 2) << 
    " BeamSearch constraint input sequences must have 2 axes," <<
    " batch_size and input_sequence_length";
}

template <typename Dtype>
void SequenceConstrainedBeamSearchLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  BaseBeamSearchLayer<Dtype>::Reshape(bottom, top);
  // Reshape temp data structures to allow for beam size + 1 expansions to rank
  const int batch_size = bottom[0]->shape(0);
  vector<int> shape;
  shape.push_back(batch_size*this->num_states_);
  shape.push_back(this->beam_size_);
  // In the constrained beam, allow for an extra expansion for one constraint word
  shape.push_back(this->beam_size_+1);
  this->score_.Reshape(shape);
  this->score_indices_.Reshape(shape);
}

template <typename Dtype>
int SequenceConstrainedBeamSearchLayer<Dtype>::NumStates(const vector<Blob<Dtype>*>& bottom) {
  const Dtype* constraint_data = bottom[0]->cpu_data();
  const int input_sequence_length = bottom[0]->shape(1);
  int max_length = 1;
  for (int n=0; n<bottom[0]->shape(0); ++n){
    for (int i=0; i<input_sequence_length; ++i){
      if (constraint_data[n*input_sequence_length+i] == Dtype(this->end_of_sequence_)){
        break;
      } else if (max_length < i+1) {
        max_length = i+1;
      }      
    }
  }
  int num_states = 1 + max_length; // Unconstrained and one constrained for each partial sequence
  DLOG(INFO) << "Using " << num_states << " constrained beam search states";
  return num_states; 
}

template <typename Dtype>
void SequenceConstrainedBeamSearchLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}


#ifdef CPU_ONLY
STUB_GPU_FORWARD(SequenceConstrainedBeamSearchLayer, Forward);
#endif

INSTANTIATE_CLASS(SequenceConstrainedBeamSearchLayer);
REGISTER_LAYER_CLASS(SequenceConstrainedBeamSearch);

}  // namespace caffe
