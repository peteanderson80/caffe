#ifndef CAFFE_LSTM_NODE_LAYER_HPP_
#define CAFFE_LSTM_NODE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Processes sequential inputs using a "Long Short-Term Memory" (LSTM)
 *        [1] style recurrent neural network (RNN). Unlike the LSTM layer, this
 *        implementation represents a single LSTM timestep, requiring the full net to
 *        be unrolled in prototxt. However, this approach enables sampling and beam
 *        search through the resulting network, which the LSTM layer cannot do.
 *
 * The specific architecture used in this implementation is as described in
 * "Learning to Execute" [2], reproduced below:
 *     @f$ i_t := \sigmoid[ W_{hi} * h_{t-1} + W_{xi} * x_t + b_i ] @f$
 *     @f$ f_t := \sigmoid[ W_{hf} * h_{t-1} + W_{xf} * x_t + b_f ] @f$
 *     @f$ o_t := \sigmoid[ W_{ho} * h_{t-1} + W_{xo} * x_t + b_o ] @f$
 *     @f$ g_t :=    \tanh[ W_{hg} * h_{t-1} + W_{xg} * x_t + b_g ] @f$
 *     @f$ c_t := (f_t .* c_{t-1}) + (i_t .* g_t) @f$
 *     @f$ h_t := o_t .* \tanh[c_t] @f$
 *
 * Notably, this implementation lacks the "diagonal" gates, as used in the
 * LSTM architectures described by Alex Graves [3] and others.
 *
 * [1] Hochreiter, Sepp, and Schmidhuber, Jürgen. "Long short-term memory."
 *     Neural Computation 9, no. 8 (1997): 1735-1780.
 *
 * [2] Zaremba, Wojciech, and Sutskever, Ilya. "Learning to execute."
 *     arXiv preprint arXiv:1410.4615 (2014).
 *
 * [3] Graves, Alex. "Generating sequences with recurrent neural networks."
 *     arXiv preprint arXiv:1308.0850 (2013).
 */
template <typename Dtype>
class LSTMNodeLayer : public Layer<Dtype> {
 public:
  explicit LSTMNodeLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "LSTMNode"; }
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MaxBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return 2; }
 protected:

/**
   * @param bottom input Blob vector (length 2 or 3)
   *   -# the inputs @f$ x_t @f$
   *   -# the previous hidden state @f$ c_{t-1} @f$
   *   -# optional continuation indicators (1, or 0 for the start of
            a new sequence)
   * @param top output Blob vector (length 2)
   *   -# the output @f$ h_t @f$
   *   -# the current hidden state @f$ c_t @f$
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int M_; // batch size
  int N_; // num memory cells
  int K_; // input data size
  Blob<Dtype> gates_data_buffer_;
  bool bias_term_;
  Blob<Dtype> ones_;
  Blob<Dtype> tanh_c_;
};


}  // namespace caffe

#endif  // CAFFE_LSTM_NODE_LAYER_HPP_
