#include <vector>
#include "caffe/filler.hpp"
#include "caffe/layers/lstm_node_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
inline Dtype sigmoid(const Dtype x) {
  return 1. / (1. + exp(-x));
}

template <typename Dtype>
inline Dtype sigmoid_diff(const Dtype x) {
  return x * (1. - x);
}

template <typename Dtype>
inline Dtype tanh_diff(const Dtype x) {
  return (1. - x * x);
}

template <typename Dtype>
void LSTMNodeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LSTMParameter lstm_param = this->layer_param_.lstm_param();
  bias_term_ = lstm_param.bias_term();
  cache_gates_ = lstm_param.cache_gates();
  N_ = lstm_param.num_cells();
  K_ = bottom[0]->count(1);

  if (bias_term_) {
    this->blobs_.resize(2);
  } else {
    this->blobs_.resize(1);
  }
  // Intialize the weights
  vector<int> weight_shape(2);
  weight_shape[0] = 4*N_; // i,f,o,g
  weight_shape[1] = K_;
  this->blobs_[0].reset(new Blob<Dtype>(weight_shape));

  // fill weights - 2 step procedure is required to initialise with
  // four different fillers, making temp use of another blob
  weight_shape[0] = N_;
  weight_shape[1] = K_;
  gates_data_buffer_.Reshape(weight_shape);
  Dtype* weight_data = this->blobs_[0]->mutable_cpu_data();

  shared_ptr<Filler<Dtype> > input_gate_weight_filler(GetFiller<Dtype>(
      lstm_param.input_gate_weight_filler()));
  input_gate_weight_filler->Fill(&gates_data_buffer_);
  caffe_copy(N_ * K_, gates_data_buffer_.cpu_data(), weight_data);
  weight_data += N_ * K_;

  shared_ptr<Filler<Dtype> > forget_gate_weight_filler(GetFiller<Dtype>(
      lstm_param.forget_gate_weight_filler()));
  forget_gate_weight_filler->Fill(&gates_data_buffer_);
  caffe_copy(N_ * K_, gates_data_buffer_.cpu_data(), weight_data);
  weight_data += N_ * K_;

  shared_ptr<Filler<Dtype> > output_gate_weight_filler(GetFiller<Dtype>(
      lstm_param.output_gate_weight_filler()));
  output_gate_weight_filler->Fill(&gates_data_buffer_);
  caffe_copy(N_ * K_, gates_data_buffer_.cpu_data(), weight_data);
  weight_data += N_ * K_;

  shared_ptr<Filler<Dtype> > input_weight_filler(GetFiller<Dtype>(
      lstm_param.input_weight_filler()));
  input_weight_filler->Fill(&gates_data_buffer_);
  caffe_copy(N_ * K_, gates_data_buffer_.cpu_data(), weight_data);

  // If necessary, intiialize and fill the bias term, again this is
  // a 2 step procedure.
  if (bias_term_) {
    vector<int> bias_shape(1, 4*N_); // 4x is for i,f,o,g
    this->blobs_[1].reset(new Blob<Dtype>(bias_shape));

    bias_shape[0] = N_;
    gates_data_buffer_.Reshape(bias_shape);
    Dtype* bias_data = this->blobs_[1]->mutable_cpu_data();

    shared_ptr<Filler<Dtype> > input_gate_bias_filler(GetFiller<Dtype>(
        lstm_param.input_gate_bias_filler()));
    input_gate_bias_filler->Fill(&gates_data_buffer_);
    caffe_copy(N_, gates_data_buffer_.cpu_data(), bias_data);
    bias_data += N_;

    shared_ptr<Filler<Dtype> > forget_gate_bias_filler(GetFiller<Dtype>(
        lstm_param.forget_gate_bias_filler()));
    forget_gate_bias_filler->Fill(&gates_data_buffer_);
    caffe_copy(N_, gates_data_buffer_.cpu_data(), bias_data);
    bias_data += N_;

    shared_ptr<Filler<Dtype> > output_gate_bias_filler(GetFiller<Dtype>(
        lstm_param.output_gate_bias_filler()));
    output_gate_bias_filler->Fill(&gates_data_buffer_);
    caffe_copy(N_, gates_data_buffer_.cpu_data(), bias_data);
    bias_data += N_;

    shared_ptr<Filler<Dtype> > input_bias_filler(GetFiller<Dtype>(
        lstm_param.input_bias_filler()));
    input_bias_filler->Fill(&gates_data_buffer_);
    caffe_copy(N_, gates_data_buffer_.cpu_data(), bias_data);
  }
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void LSTMNodeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_K = bottom[0]->count(1);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with lstm parameters.";
  M_ = bottom[0]->count(0,1);
  vector<int> weight_shape(2);
  weight_shape[0] = M_;
  weight_shape[1] = N_;
  tanh_c_.Reshape(weight_shape);
  top[0]->Reshape(weight_shape);
  top[1]->Reshape(weight_shape);
  weight_shape[1] = 4 * N_;
  gates_data_buffer_.Reshape(weight_shape);
  CHECK_EQ(bottom[1]->count(), top[1]->count())
      << "Input mem cells M * N size.";
  if (bottom.size() > 2) {
    CHECK_EQ(bottom[2]->count(), M_)
      << "Continuation indicator blob must have count equal to num";
  }
  // Set up the bias multiplier / cont indicators
  vector<int> mult_shape(1, M_);
  ones_.Reshape(mult_shape);
  caffe_set(M_, Dtype(1), ones_.mutable_cpu_data());
}

template <typename Dtype>
void LSTMNodeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  // Continue same series or not
  const Dtype* cont = ones_.cpu_data();
  if (bottom.size() > 2) {
    cont = bottom[2]->cpu_data();
  }

  // Perform weight and bias updates
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* gates_data = gates_data_buffer_.mutable_cpu_data();
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, 4*N_, K_, (Dtype)1.,
      bottom_data, weight, (Dtype)0., gates_data);
  if (bias_term_) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, 4*N_, 1, (Dtype)1.,
        ones_.cpu_data(),
        this->blobs_[1]->cpu_data(), (Dtype)1., gates_data);
  }

  // Apply activation functions and calc output
  Dtype* i = gates_data + N_ * 0;
  Dtype* f = gates_data + N_ * 1;
  Dtype* o = gates_data + N_ * 2;
  Dtype* g = gates_data + N_ * 3;
  Dtype* tanh_c = tanh_c_.mutable_cpu_data();
  const Dtype* c_prev = bottom[1]->cpu_data();
  Dtype* h = top[0]->mutable_cpu_data();
  Dtype* c = top[1]->mutable_cpu_data();
  for (int n = 0; n < M_; ++n) {
    for (int j = 0; j < N_; ++j) {
      const int idx = j + n * N_;
      const int gdx = j + n * 4 * N_; // gates_data index
      i[gdx] = sigmoid(i[gdx]);
      f[gdx] = sigmoid(f[gdx]) * cont[n];
      o[gdx] = sigmoid(o[gdx]);
      g[gdx] = tanh(g[gdx]);
      c[idx] = f[gdx] * c_prev[idx] + i[gdx] * g[gdx];
      tanh_c[idx] = tanh(c[idx]);
      h[idx] = o[gdx] * tanh_c[idx];
    }
  }
}

template <typename Dtype>
void LSTMNodeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  if (!cache_gates_){
    Forward_cpu(bottom, top);
  }

  const Dtype* h_diff = top[0]->cpu_diff();
  const Dtype* c_diff = top[1]->cpu_diff();

  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* c_prev = bottom[1]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  Dtype* c_prev_diff = bottom[1]->mutable_cpu_diff();

  const Dtype* gates_data = gates_data_buffer_.mutable_cpu_data();
  const Dtype* i = gates_data + N_ * 0;
  const Dtype* f = gates_data + N_ * 1;
  const Dtype* o = gates_data + N_ * 2;
  const Dtype* g = gates_data + N_ * 3;
  const Dtype* tanh_c = tanh_c_.cpu_data();

  Dtype* gates_diff = gates_data_buffer_.mutable_cpu_diff();
  Dtype* i_diff = gates_diff + N_ * 0;
  Dtype* f_diff = gates_diff + N_ * 1;
  Dtype* o_diff = gates_diff + N_ * 2;
  Dtype* g_diff = gates_diff + N_ * 3;

  for (int n = 0; n < M_; ++n) {
    for (int j = 0; j < N_; ++j) {
      const int idx = j + n * N_;
      const int gdx = j + n * 4 * N_; // gates_data index
      const Dtype c_term_diff = c_diff[idx] + h_diff[idx] * o[gdx] * tanh_diff(tanh_c[idx]);
      if (propagate_down[1]) {
        // gradient with respect to previous hidden state
        c_prev_diff[idx] = c_term_diff * f[gdx];
      }
      i_diff[gdx] = c_term_diff * g[gdx] * sigmoid_diff(i[gdx]);
      f_diff[gdx] = c_term_diff * c_prev[idx] * sigmoid_diff(f[gdx]);
      o_diff[gdx] = h_diff[idx] * tanh_c[idx] * sigmoid_diff(o[gdx]);
      g_diff[gdx] = c_term_diff * i[gdx] * tanh_diff(g[gdx]);
    }
  }
  if (this->param_propagate_down_[0]) {
    // Gradient with respect to weight
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, 4*N_, K_, M_, (Dtype)1.,
        gates_diff, bottom_data, (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    // Gradient with respect to bias
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, 4*N_, (Dtype)1., gates_diff,
        ones_.cpu_data(), (Dtype)1., this->blobs_[1]->mutable_cpu_diff());
  }
  if (propagate_down[0]) {
    // Gradient with respect to bottom data
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, 4*N_, (Dtype)1.,
        gates_diff, this->blobs_[0]->cpu_data(), (Dtype)0., bottom_diff);
  }
}


#ifdef CPU_ONLY
STUB_GPU(LSTMNodeLayer);
#endif

INSTANTIATE_CLASS(LSTMNodeLayer);
REGISTER_LAYER_CLASS(LSTMNode);

}  // namespace caffe
