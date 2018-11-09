#include <cmath>

#include "caffe/layers/constrained_beam_search_layer.hpp"


namespace caffe {


template <typename Dtype>
void ConstrainedBeamSearchLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  BaseBeamSearchLayer<Dtype>::LayerSetUp(bottom, top);
  // Check the constraint inputs are ok
  CHECK_EQ(bottom[0]->num_axes(), 3) << 
    " BeamSearch constraint input must have 3 axes, batch_size, num_conjunctions and num_disjunctions";
}

template <typename Dtype>
void ConstrainedBeamSearchLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  BaseBeamSearchLayer<Dtype>::Reshape(bottom, top);
  // Reshape temp data structures to allow for bag_size*num_alternatives + beam size expansions to rank
  const int batch_size = bottom[0]->shape(0);
  const int bag_size = bottom[0]->shape(1);
  const int num_alternatives = bottom[0]->shape(2);
  vector<int> shape;
  shape.push_back(batch_size*this->num_states_);
  shape.push_back(this->beam_size_);
  // In the constrained beam, allow for expansions for each constraint word in the bag
  shape.push_back(this->beam_size_+bag_size*num_alternatives);
  this->score_.Reshape(shape);
  this->score_indices_.Reshape(shape);
}

template <typename Dtype>
int ConstrainedBeamSearchLayer<Dtype>::NumStates(const vector<Blob<Dtype>*>& bottom) {
  const int bag_size = bottom[0]->shape(1);
  int num_states = std::pow(2, bag_size); // Unconstrained and constrained for each supplied constraint word
  DLOG(INFO) << "Using " << num_states << " constrained beam search states";
  return num_states; 
}

template <typename Dtype>
void ConstrainedBeamSearchLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}


#ifdef CPU_ONLY
STUB_GPU_FORWARD(ConstrainedBeamSearchLayer, Forward);
#endif

INSTANTIATE_CLASS(ConstrainedBeamSearchLayer);
REGISTER_LAYER_CLASS(ConstrainedBeamSearch);

}  // namespace caffe
