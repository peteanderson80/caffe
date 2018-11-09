#ifndef CAFFE_SEQUENCE_CONSTRAINED_BEAM_SEARCH_LAYER_HPP_
#define CAFFE_SEQUENCE_CONSTRAINED_BEAM_SEARCH_LAYER_HPP_

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
 *        In this implementation, the constraint takes the form of a single word sequence.
 *        A single timestep of the RNN should be defined in a proto file and provided as
 *        a parameter, along with the beam size, a list of recurrent connections, etc.
 * @param bottom input Blob vector (length 2+)
 *   -# @f$ (N \times L) @f$
 *        the sequences of maximum length @f$ L @f$ that must be included in the Beam
 *        Search output.
 *   -# @f$ (N \times X1 \times X2 \times X3) @f$
 *        the non-recurrent inputs to the RNN (e.g. data, CNN output, etc)
 * @param top output Blob vector (length 3)
 *   -# @f$ (N \times (L+1) \times B \times T) @f$ 
 *        sequences of output indices of (maximum) length @f$ T @f$, ordered by 
 *        decreasing summed scores, for each beam @f$ B @f$ for each
 *        additional constraint @f$ (1+C) @f$.
 *   -# @f$ (N \times (L+1) \times B) @f$ 
 *        the summed score for each beam sequence @f$ B @f$.
 *   -# @f$ (N \times (L+1) \times B \times T) @f$ 
 *        output scores of (maximum) length @f$ T @f$ for each beam @f$ B @f$ for each
 *        additional input constraint @f$ (L+1) @f$.
 */
template <typename Dtype>
class SequenceConstrainedBeamSearchLayer : public BaseBeamSearchLayer<Dtype> {
 public:
  explicit SequenceConstrainedBeamSearchLayer(const LayerParameter& param)
      : BaseBeamSearchLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SequenceConstrainedBeamSearch"; }
  virtual inline int MinBottomBlobs() const { return 2; } // Constraints come first

 protected:

  virtual int NumStates(const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);


};

}  // namespace caffe

#endif  // CAFFE_SEQUENCE_CONSTRAINED_BEAM_SEARCH_LAYER_HPP_
