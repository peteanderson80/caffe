#include <vector>
#include "caffe/layers/lstm_node_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__device__ Dtype cuda_sigmoid(const Dtype x) {
  return 1. / (1. + exp(-x));
}

template <typename Dtype>
__device__ Dtype cuda_sigmoid_diff(const Dtype x) {
  return x * (1. - x);
}

template <typename Dtype>
__device__ Dtype cuda_tanh_diff(const Dtype x) {
  return (1. - x * x);
}

template <typename Dtype>
__global__ void ForwardCombineGates(
  const int nthreads,
  const int N_,
  const Dtype* cont,
  const Dtype* c_prev,
  Dtype* i,
  Dtype* f,
  Dtype* o,
  Dtype* g,
  Dtype* c,
  Dtype* tanh_c,
  Dtype* h)
{
  CUDA_KERNEL_LOOP(idx, nthreads) {
    const int n = idx/N_;
    const int j = idx%N_;
    const int gdx = j + n * 4 * N_; // gates_data index
    i[gdx] = cuda_sigmoid(i[gdx]);
    f[gdx] = cuda_sigmoid(f[gdx]) * cont[n];
    o[gdx] = cuda_sigmoid(o[gdx]);
    g[gdx] = tanh(g[gdx]);
    c[idx] = f[gdx] * c_prev[idx] + i[gdx] * g[gdx];
    tanh_c[idx] = tanh(c[idx]);
    h[idx] = o[gdx] * tanh_c[idx];
  }
}

template <typename Dtype>
void LSTMNodeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  // Continue same series or not
  const Dtype* cont = ones_.gpu_data();
  if (bottom.size() > 2) {
    cont = bottom[2]->gpu_data();
  }

  // Perform weight and bias updates
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* gates_data = gates_data_buffer_.mutable_gpu_data();
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, 4*N_, K_, (Dtype)1.,
      bottom_data, weight, (Dtype)0., gates_data);
  if (bias_term_) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, 4*N_, 1, (Dtype)1.,
        ones_.gpu_data(),
        this->blobs_[1]->gpu_data(), (Dtype)1., gates_data);
  }

  // Apply activation functions and calc output
  Dtype* i = gates_data + N_ * 0;
  Dtype* f = gates_data + N_ * 1;
  Dtype* o = gates_data + N_ * 2;
  Dtype* g = gates_data + N_ * 3;
  Dtype* tanh_c = tanh_c_.mutable_gpu_data();
  const Dtype* c_prev = bottom[1]->gpu_data();
  Dtype* h = top[0]->mutable_gpu_data();
  Dtype* c = top[1]->mutable_gpu_data();
  const int count = M_ * N_;
  ForwardCombineGates<Dtype> // NOLINT_NEXT_LINE(whitespace/operators)
  <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, N_, cont, c_prev, i, f, o, g, c, tanh_c, h);
  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
__global__ void BackwardGates(
  const int nthreads,
  const int N_,
  const Dtype* i,
  const Dtype* f,
  const Dtype* o,
  const Dtype* g,
  const Dtype* tanh_c,
  const Dtype* c_prev,
  Dtype* i_diff,
  Dtype* f_diff,
  Dtype* o_diff,
  Dtype* g_diff,
  Dtype* c_prev_diff,
  const Dtype* h_diff,
  const Dtype* c_diff)
{
  CUDA_KERNEL_LOOP(idx, nthreads) {
    const int n = idx/N_;
    const int j = idx%N_;
    const int gdx = j + n * 4 * N_; // gates_data index
    const Dtype c_term_diff = c_diff[idx] + h_diff[idx] * o[gdx] * cuda_tanh_diff(tanh_c[idx]);
    c_prev_diff[idx] = c_term_diff * f[gdx];
    i_diff[gdx] = c_term_diff * g[gdx] * cuda_sigmoid_diff(i[gdx]);
    f_diff[gdx] = c_term_diff * c_prev[idx] * cuda_sigmoid_diff(f[gdx]);
    o_diff[gdx] = h_diff[idx] * tanh_c[idx] * cuda_sigmoid_diff(o[gdx]);
    g_diff[gdx] = c_term_diff * i[gdx] * cuda_tanh_diff(g[gdx]);

  }
}

template <typename Dtype>
void LSTMNodeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  if (!cache_gates_){
    Forward_gpu(bottom, top);
  }

  const Dtype* h_diff = top[0]->gpu_diff();
  const Dtype* c_diff = top[1]->gpu_diff();

  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* c_prev = bottom[1]->gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  Dtype* c_prev_diff = bottom[1]->mutable_gpu_diff();

  const Dtype* gates_data = gates_data_buffer_.mutable_gpu_data();
  const Dtype* i = gates_data + N_ * 0;
  const Dtype* f = gates_data + N_ * 1;
  const Dtype* o = gates_data + N_ * 2;
  const Dtype* g = gates_data + N_ * 3;
  const Dtype* tanh_c = tanh_c = tanh_c_.gpu_data();

  Dtype* gates_diff = gates_data_buffer_.mutable_gpu_diff();
  Dtype* i_diff = gates_diff + N_ * 0;
  Dtype* f_diff = gates_diff + N_ * 1;
  Dtype* o_diff = gates_diff + N_ * 2;
  Dtype* g_diff = gates_diff + N_ * 3;

  const int count = M_ * N_;
  BackwardGates<Dtype> // NOLINT_NEXT_LINE(whitespace/operators)
  <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, N_, i, f, o, g, tanh_c, c_prev, i_diff, f_diff, o_diff, g_diff,
          c_prev_diff, h_diff, c_diff);
  CUDA_POST_KERNEL_CHECK;

  if (this->param_propagate_down_[0]) {
    // Gradient with respect to weight
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, 4*N_, K_, M_, (Dtype)1.,
        gates_diff, bottom_data, (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    // Gradient with respect to bias
    caffe_gpu_gemv<Dtype>(CblasTrans, M_, 4*N_, (Dtype)1., gates_diff,
        ones_.gpu_data(), (Dtype)1., this->blobs_[1]->mutable_gpu_diff());
  }
  if (propagate_down[0]) {
    // Gradient with respect to bottom data
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, 4*N_, (Dtype)1.,
        gates_diff, this->blobs_[0]->gpu_data(), (Dtype)0., bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(LSTMNodeLayer);

}  // namespace caffe
