#ifndef CAFFE_CONSTRAINED_BEAM_SEARCH_LAYER_HPP_
#define CAFFE_CONSTRAINED_BEAM_SEARCH_LAYER_HPP_

#include <vector>
#include <queue>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_beam_search_layer.hpp"

namespace caffe {

/**
 * @brief Performs Constrained Beam Search decoding of "Long Short-Term Memory" (LSTM)
 *        style recurrent neural networks (RNNs) using the LSTMNode layer implementation.
 *        In this implementation, constraints take the form of conjunctions of disjunctions.
 *        A single timestep of the RNN should be defined in a proto file and provided as
 *        a parameter, along with the beam size, a list of recurrent connections, etc.
 * @param bottom input Blob vector (length 2+)
 *   -# @f$ (N \times C \times D) @f$
 *        the @f$ C @f$ conjuctions of @f$ D @f$ disjunctions representing the 
 *        Beam Search constraints
 *   -# @f$ (N \times X1 \times X2 \times X3) @f$
 *        the non-recurrent inputs to the RNN (e.g. data, CNN output, etc)
 * @param top output Blob vector (length 3)
 *   -# @f$ (N \times 2^C \times B \times T) @f$
 *        sequences of output indices of (maximum) length @f$ T @f$, ordered by 
 *        decreasing summed scores, for each beam @f$ B @f$ for each
 *        possible constraint combination @f$ 2^C @f$.
 *   -# @f$ (N \times 2^C \times B) @f$ 
 *        the summed score for each beam sequence @f$ B @f$.
 *   -# @f$ (N \times 2^C \times B \times T) @f$ 
 *        output scores of (maximum) length @f$ T @f$ for each beam @f$ B @f$ for each
 *        possible constraint combination @f$ 2^C @f$.
 */
template <typename Dtype>
class ConstrainedBeamSearchLayer : public BaseBeamSearchLayer<Dtype> {
 public:
  explicit ConstrainedBeamSearchLayer(const LayerParameter& param)
      : BaseBeamSearchLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ConstrainedBeamSearch"; }
  virtual inline int MinBottomBlobs() const { return 2; } // Constraints come first

 protected:

  virtual int NumStates(const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);


};

}  // namespace caffe

#endif  // CAFFE_CONSTRAINED_BEAM_SEARCH_LAYER_HPP_
