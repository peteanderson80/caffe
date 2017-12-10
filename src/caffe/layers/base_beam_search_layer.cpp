#include <algorithm>
#include <vector>
#include <boost/concept_check.hpp>

#include "caffe/layers/base_beam_search_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"

namespace caffe {

template <typename Dtype>
void BaseBeamSearchLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const BeamSearchParameter& beam_search_param = this->layer_param_.beam_search_param();
  sequence_length_ = beam_search_param.sequence_length();
  beam_size_ = beam_search_param.beam_size();
  end_of_sequence_ = beam_search_param.end_of_sequence();
  prevent_repeats_ = beam_search_param.prevent_repeats();
  allowed_multiple_size_ = beam_search_param.allowed_multiple_size();
  if (allowed_multiple_size_ > 0) {
    LOG(INFO) << "Output tokens will be restricted from appearing multiple times.";
    std::vector<int> shape(1, allowed_multiple_size_);
    allowed_multiple_.Reshape(shape);
    Dtype* data = allowed_multiple_.mutable_cpu_data();
    for (int i=0; i< allowed_multiple_size_; ++i) {
      data[i] = Dtype(beam_search_param.allowed_multiple(i));
      LOG(INFO) << "Allowed multiple " << data[i];
    }
  }
  // Create the beam search net.
  NetParameter net_param;
  if (beam_search_param.has_net_param()) {
    LOG(INFO) << "Creating beam search net specified in net_param.";
    net_param.CopyFrom(beam_search_param.net_param());
  }
  if (beam_search_param.has_net()) {
    LOG(INFO) << "Creating beam search net from net file: " << beam_search_param.net();
    ReadNetParamsFromTextFileOrDie(beam_search_param.net(), &net_param);
  }
  net_.reset(new Net<Dtype>(net_param));
  // Load pretrained weights directly (if provided)
  if (beam_search_param.has_weights()) {
    net_->CopyTrainedLayersFrom(beam_search_param.weights());
  }

  // This layer's parameters are any parameters in the layers of the beam search
  // net. We only want one copy of each parameter, so check that the parameter
  // is "owned" by the layer, rather than shared with another.
  this->blobs_.clear();
  for (int i = 0; i < net_->params().size(); ++i) {
    if (net_->param_owners()[i] == -1) {
      LOG(INFO) << "Adding parameter " << i << ": "
                << net_->param_display_names()[i];
      this->blobs_.push_back(net_->params()[i]);
    }
  }
  // Set param_propagate_down to false in this layer.
  this->param_propagate_down_.clear();
  this->param_propagate_down_.resize(this->blobs_.size(), false);
  // Keep track of recurrent connections in the beam search net
  string from = beam_search_param.beam_search_connection().src();
  string to = beam_search_param.beam_search_connection().dest();
  CHECK(net_->has_blob(from)) << from << " blob not found in BeamSearch net_param";
  CHECK(net_->has_blob(to)) << to << " blob not found in BeamSearch net_param";
  output_score_ = net_->blob_by_name(from);
  vocab_size_ = output_score_->shape(-1);
  input_sequence_ = net_->blob_by_name(to);
  CHECK_EQ(input_sequence_->num_axes(), 2) << " BeamSearch 'to' connection must have 2 axes";
  CHECK_EQ(input_sequence_->shape(1), 1) << " BeamSearch 'to' connection must have shape (batch_size, 1)";
  for (int i = 0; i < beam_search_param.recurrent_connection_size(); ++i) {
    from = beam_search_param.recurrent_connection(i).src();
    to = beam_search_param.recurrent_connection(i).dest();
    CHECK(net_->has_blob(from)) << from << " blob not found in BeamSearch net_param";
    CHECK(net_->has_blob(to)) << to << " blob not found in BeamSearch net_param";
    shared_ptr<Blob<Dtype> > from_blob = net_->blob_by_name(from);
    shared_ptr<Blob<Dtype> > to_blob = net_->blob_by_name(to);
    CHECK_EQ(from_blob->num_axes(), to_blob->num_axes()) << from << " and "
        << to << " have different num_axes";
    for (int j = 0; j < from_blob->num_axes(); ++j) {
      CHECK_EQ(from_blob->shape(j), to_blob->shape(j)) << from << " and "
        << to << " have different shapes";
    }
    recurrent_connections_.push_back(std::make_pair(from_blob, to_blob));
  }
}

template <typename Dtype>
void BaseBeamSearchLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const BeamSearchParameter& beam_search_param = this->layer_param_.beam_search_param();
  // Check for consistent bottom batch sizes
  const int batch_size = bottom[0]->shape(0);
  for (int i=1; i<bottom.size(); ++i) {
    CHECK_EQ(batch_size, bottom[i]->shape(0))
      << "bottom blobs must have equal batch sizes).";
  }
  // Reshape any recurrent input layers within the beam search net to have correct batch size
  vector<int> input_shape(input_sequence_->shape());
  num_stratas_ = ExactNumStratas(bottom);
  input_shape[0] = batch_size * beam_size_ * num_stratas_;
  input_sequence_->Reshape(input_shape);
  DLOG(INFO) << "Reshaped BeamSearch net " << beam_search_param.beam_search_connection().dest()
      << " to " << input_sequence_->shape_string();
  for (int i = 0; i < recurrent_connections_.size(); ++i) {
    vector<int> shape(recurrent_connections_[i].second->shape());
    shape[0] = batch_size * beam_size_ * num_stratas_;
    recurrent_connections_[i].second->Reshape(shape);
    DLOG(INFO) << "Reshaped BeamSearch net " << beam_search_param.recurrent_connection(i).dest()
    << " to " << recurrent_connections_[i].second->shape_string();
  }
  // Reshape inputs from bottom blobs - bottom blobs are transferred in by position, not name
  for (int i = 0; i < bottom.size(); ++i) {
    vector<int> bottom_shape = net_->input_blobs().at(i)->shape();
    bottom_shape[0] = batch_size * beam_size_;
    net_->input_blobs().at(i)->Reshape(bottom_shape);
  }
  net_->Reshape();
  // Setup temp data structures and reshape output
  output_score_indices_.Reshape(output_score_->shape());
  vector<int> shape;
  shape.push_back(batch_size*num_stratas_);
  shape.push_back(beam_size_);
  shape.push_back(beam_size_);
  score_.Reshape(shape);
  score_indices_.Reshape(shape);
  shape.clear();
  shape.push_back(batch_size);
  shape.push_back(num_stratas_);
  shape.push_back(beam_size_);
  top[1]->Reshape(shape); // summed scores (batch_size, num_stratas, beam_size)
  shape.push_back(sequence_length_);
  top[0]->Reshape(shape); // sequences (batch_size, num_stratas, beam_size, sequence_length)
  top[2]->Reshape(shape); // score sequence (batch_size, num_stratas, beam_size, sequence_length)
}

template <typename Dtype>
void BaseBeamSearchLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i=0; i<propagate_down.size(); ++i) {
    if (propagate_down[i]) {
      LOG(FATAL) << this->type() << " Layer cannot backpropagate.";
    }
  }
}

template <typename Dtype>
void BaseBeamSearchLayer<Dtype>::copy_bottom_inputs_cpu(const vector<Blob<Dtype>*>& bottom) {
  // Copy bottom input into the beam search net, duplicating for size of beam search
  const int batch_size = bottom[0]->shape(0);
  for (int i = 0; i < bottom.size(); ++i) {
    string blob_name = this->layer_param().bottom(i);
    if (this->net_->has_blob(blob_name)){
      const Dtype* src_data = bottom[i]->cpu_data();
      Dtype* tgt_data = this->net_->blob_by_name(blob_name)->mutable_cpu_data();
      for (int k = 0; k < batch_size; ++k) {
        for (int j = 0; j < this->beam_size_; ++j) { // Multiple copies to accomodate beam size
          caffe_copy(bottom[i]->count()/batch_size, src_data, tgt_data);
          tgt_data += bottom[i]->count()/batch_size;
        }
        src_data += bottom[i]->count()/batch_size;
      }
      DLOG(INFO) << "Copied " << blob_name << " into BeamSearch net";
    }
  }
}

template <typename Dtype>
void BaseBeamSearchLayer<Dtype>::clear_recurrent_inputs_cpu() {
  caffe_set(this->input_sequence_->count(), Dtype(this->end_of_sequence_), this->input_sequence_->mutable_cpu_data());
  for (int i = 0; i < this->recurrent_connections_.size(); ++i) {
    caffe_set(this->recurrent_connections_[i].second->count(), Dtype(0),
              this->recurrent_connections_[i].second->mutable_cpu_data());
  }
}


INSTANTIATE_CLASS(BaseBeamSearchLayer);

}  // namespace caffe
