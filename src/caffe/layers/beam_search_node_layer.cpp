#include <vector>

#include "caffe/layers/beam_search_node_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BeamSearchNodeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const BeamSearchParameter& beam_search_param = this->layer_param_.beam_search_param();
  beam_size_ = beam_search_param.beam_size();
  end_of_sequence_ = beam_search_param.end_of_sequence();
  ignore_label_ = beam_search_param.ignore_label();
  CHECK(bottom.size() == top.size() || bottom.size()+2 == top.size())
    << "top blob count must be equal or two greater than bottom blob count.";
  has_input_scores_ = (bottom.size() == top.size());
  for (int i=0; i<bottom.size(); ++i) {
    CHECK_EQ(bottom[i]->num_axes(), 2)
        << "all inputs must have two axes.";
  }
  prevent_repeats_ = beam_search_param.prevent_repeats();
  allowed_multiple_size_ = beam_search_param.allowed_multiple_size();
  if (allowed_multiple_size_ > 0) {
    DLOG(INFO) << "Output tokens will be restricted from appearing multiple times.";
    std::vector<int> shape(1, allowed_multiple_size_);
    allowed_multiple_.Reshape(shape);
    Dtype* data = allowed_multiple_.mutable_cpu_data();
    for (int i=0; i< allowed_multiple_size_; ++i) {
      data[i] = Dtype(beam_search_param.allowed_multiple(i));
      DLOG(INFO) << "Allowed multiple " << data[i];
    }
  }
}

template <typename Dtype>
void BeamSearchNodeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (has_input_scores_) {
    vocab_size_ = bottom[2]->shape(1);
    exp_.Reshape(bottom[2]->shape());
    exp_indices_.Reshape(bottom[2]->shape());
  } else {
    vocab_size_ = bottom[0]->shape(1);
    exp_.Reshape(bottom[0]->shape());
    exp_indices_.Reshape(bottom[0]->shape());
  }
  batch_size_ = bottom[0]->shape(0);
  timestep_ = has_input_scores_ ? bottom[1]->shape(1) : 0;
  for (int i = 0; i < (has_input_scores_ ? 3 : 1); ++i) {
    CHECK_EQ(bottom[i]->shape(0), batch_size_)
        << "inputs must have the same batch size.";
  }
  vector<int> shape(2, 1);
  shape[0] = batch_size_;
  top[0]->Reshape(shape);
  top[2]->Reshape(shape);
  source_map_.Reshape(shape);
  shape[1] = timestep_ + 1;
  top[1]->Reshape(shape);
  for (int i=NonRecurrentOutputs(); i<top.size(); ++i) {
    if (has_input_scores_){
      top.at(i)->ReshapeLike(*(bottom.at(i)));
    } else {
      top.at(i)->ReshapeLike(*(bottom.at(i-2)));
    }
  }
  shape[1] = beam_size_;
  score_.Reshape(shape);
  score_indices_.Reshape(shape);
}

template <typename Dtype>
void BeamSearchNodeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* partial_sum = NULL;
  const Dtype* seq = NULL;
  const Dtype* score = NULL;
  if (has_input_scores_) {
    partial_sum = bottom[0]->cpu_data();
    seq = bottom[1]->cpu_data();
    score = bottom[2]->cpu_data();
  } else {
    score = bottom[0]->cpu_data();
  }
  Dtype* next_partial_sum = top[0]->mutable_cpu_data();
  Dtype* next_seq = top[1]->mutable_cpu_data();
  Dtype* next_input = top[2]->mutable_cpu_data();
  int* source_map_data = source_map_.mutable_cpu_data();
  for (int n = 0; n < batch_size_/beam_size_; ++n) {
    cpu_beams_.clear();
    for (int b = 0; b < beam_size_; ++b) {
      const int src = n*beam_size_+b;
      int ix = src*vocab_size_;
      if (seq && seq[src*timestep_+timestep_-1] == end_of_sequence_){
        // Keep, but don't expand a completed beam
        cpu_beams_.push_back(make_pair<Dtype, int>(partial_sum[src], ix + end_of_sequence_));
      } else {
        for (int w = 0; w < vocab_size_; ++w) {
          if (has_input_scores_) {
            if (prevent_repeats_ && seq && Dtype(w) == seq[src*timestep_+timestep_-1]){
              continue;
            }
            else {
              cpu_beams_.push_back(make_pair<Dtype, int>(partial_sum[src] + score[ix+w], ix+w));
            }
          } else {
            cpu_beams_.push_back(make_pair<Dtype,int>(score[ix+w],ix+w));
          }
        }
        if (!has_input_scores_){
          break; // On first step, only expand the first beam to prevent duplicates
        }
      }
    }
    partial_sort(cpu_beams_.begin(), cpu_beams_.begin()+beam_size_, 
        cpu_beams_.end(), std::greater<pair<Dtype,int> >());
    for (int b = 0; b < beam_size_; ++b) {
      int src = cpu_beams_.at(b).second / vocab_size_;
      Dtype word = Dtype(cpu_beams_.at(b).second % vocab_size_);
      const int dst = n*beam_size_+b;
      Dtype sum = cpu_beams_.at(b).first;
      next_input[dst] = word;
      source_map_data[dst] = src;
      for (int t = 0; t < timestep_; ++t){
        next_seq[dst*(timestep_+1)+t] = seq[src*timestep_+t];
      }
      next_seq[dst*(timestep_+1)+timestep_] = word;
      next_partial_sum[dst] = sum;
      if (word != end_of_sequence_) {
        // Copy over recurrent data to next time step
        for(int i = NonRecurrentOutputs(); i < top.size(); ++i) {
          int bottom_ix = has_input_scores_ ? i : i-2;
          caffe_copy(top[i]->shape(1), 
              bottom[bottom_ix]->cpu_data()+bottom[bottom_ix]->offset(src),
              top[i]->mutable_cpu_data()+top[i]->offset(dst));
        }
      }
    }      
  }
}

template <typename Dtype>
void BeamSearchNodeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  CHECK(!(has_input_scores_ && propagate_down[1])) 
        << "cannot backprop to partial sequence inputs.";
  const int* source_map_data = source_map_.cpu_data();
  const Dtype* next_partial_sum_diff = top[0]->cpu_diff();
  const Dtype* next_input = top[2]->cpu_data();
  Dtype* partial_sum_diff = NULL;
  Dtype* score_diff = NULL;
  const Dtype* seq = NULL;
  if (has_input_scores_) {
    partial_sum_diff = bottom[0]->mutable_cpu_diff();
    caffe_set(bottom[0]->count(), Dtype(0), partial_sum_diff);
    score_diff = bottom[2]->mutable_cpu_diff();
    caffe_set(bottom[2]->count(), Dtype(0), score_diff);
    seq = bottom[1]->cpu_data();
  } else {
    score_diff = bottom[0]->mutable_cpu_diff();
    caffe_set(bottom[2]->count(), Dtype(0), score_diff);
  }
  for(int i=3; i<top.size(); ++i){
    if (propagate_down[i]) {
      int bottom_ix = has_input_scores_ ? i : i-2;
      caffe_set(bottom[bottom_ix]->count(), Dtype(0), 
            bottom[bottom_ix]->mutable_cpu_diff());
    }
  }
  for (int n = 0; n < batch_size_/beam_size_; ++n) {
    for (int b = 0; b < beam_size_; ++b) {
      const int dst = n*beam_size_+b;
      const int src = source_map_data[dst];
      Dtype diff = next_partial_sum_diff[dst];
      // sum of scores and current score inputs
      if (has_input_scores_ && propagate_down[0]) {
        partial_sum_diff[src] += diff;
      }
      if ((has_input_scores_ && propagate_down[2]) || 
            (!has_input_scores_ && propagate_down[0])){
        if (!seq || seq[src*timestep_+timestep_-1] != end_of_sequence_){
          // Sequence has not ended
          score_diff[src*vocab_size_+int(next_input[dst])] += diff;
        }
      }
      // Recurrent connections
      for(int i=NonRecurrentOutputs(); i<top.size(); ++i){
        if (propagate_down[i]) {
          if (!seq || seq[src*timestep_+timestep_-1] != end_of_sequence_){
            // Sequence has not ended
            int bottom_ix = has_input_scores_ ? i : i-2;
            for (int j=0; j<top[i]->shape(1); ++j){
              bottom[bottom_ix]->mutable_cpu_diff()[bottom[bottom_ix]->offset(src)+j] +=
                top[i]->cpu_diff()[top[i]->offset(dst)+j];
            }
          }
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(BeamSearchNodeLayer);
#endif

INSTANTIATE_CLASS(BeamSearchNodeLayer);
REGISTER_LAYER_CLASS(BeamSearchNode);

}  // namespace caffe
