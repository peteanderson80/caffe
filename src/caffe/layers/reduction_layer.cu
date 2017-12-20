#include <vector>

#include "caffe/layers/reduction_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ReductionLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* mult_data = NULL;  Dtype* top_data = NULL;
  switch (op_) {
  case ReductionParameter_ReductionOp_SUM:
  case ReductionParameter_ReductionOp_MEAN:
    top_data = top[0]->mutable_gpu_data();
    mult_data = sum_multiplier_.gpu_data();
    caffe_gpu_gemv<Dtype>(CblasNoTrans, num_, dim_, (Dtype)1.,
                         bottom_data, mult_data, (Dtype)0., top_data);
    break;
  case ReductionParameter_ReductionOp_ASUM:
    top_data = top[0]->mutable_cpu_data();
    for (int i = 0; i < num_; ++i) {
      caffe_gpu_asum(dim_, bottom_data, top_data);
      bottom_data += dim_;
      ++top_data;
    }
    break;
  case ReductionParameter_ReductionOp_SUMSQ:
    top_data = top[0]->mutable_cpu_data();
    for (int i = 0; i < num_; ++i) {
      caffe_gpu_dot(dim_, bottom_data, bottom_data, top_data);
      bottom_data += dim_;
      ++top_data;
    }
    break;
  default:
    LOG(FATAL) << "Unknown reduction op: "
        << ReductionParameter_ReductionOp_Name(op_);
  }
  if (coeff_ != Dtype(1)) {
    // Reset the top_data pointer.
    top_data = top[0]->mutable_gpu_data();
    caffe_gpu_scal(num_, coeff_, top_data);
  }
}

template <typename Dtype>
__global__ void SetMemoryBlocks(
  const int nthreads,
  const int N,
  const Dtype* alpha, 
  const Dtype mult,
  Dtype* y)
{
  CUDA_KERNEL_LOOP(idx, nthreads) {
    for (int i=0; i<N; ++i){
      y[idx*N+i] = mult * alpha[idx];
    }
  }
}

template <typename Dtype>
void ReductionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
  const Dtype* bottom_data = NULL;
  const Dtype* top_diff = NULL;
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  switch (op_) {
  case ReductionParameter_ReductionOp_SUM:
  case ReductionParameter_ReductionOp_MEAN:
    top_diff = top[0]->gpu_diff();
    SetMemoryBlocks<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
    <<<CAFFE_GET_BLOCKS(num_), CAFFE_CUDA_NUM_THREADS>>>(
      num_, dim_, top_diff, coeff_, bottom_diff);
    break;
  case ReductionParameter_ReductionOp_ASUM:
    bottom_data = bottom[0]->gpu_data();
    top_diff = top[0]->cpu_diff();
    for (int i = 0; i < num_; ++i) {
      const Dtype bottom_coeff = (*top_diff) * coeff_;
      caffe_gpu_sign(dim_, bottom_data, bottom_diff);
      caffe_gpu_scal(dim_, bottom_coeff, bottom_diff);
      bottom_data += dim_;
      bottom_diff += dim_;
      ++top_diff;
    }
    break;
  case ReductionParameter_ReductionOp_SUMSQ:
    bottom_data = bottom[0]->gpu_data();
    top_diff = top[0]->cpu_diff();
    for (int i = 0; i < num_; ++i) {
      const Dtype bottom_coeff = (*top_diff) * coeff_;
      caffe_gpu_scale(dim_, 2 * bottom_coeff, bottom_data, bottom_diff);
      bottom_data += dim_;
      bottom_diff += dim_;
      ++top_diff;
    }
    break;
  default:
    LOG(FATAL) << "Unknown reduction op: "
        << ReductionParameter_ReductionOp_Name(op_);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ReductionLayer);

}  // namespace caffe
