#ifndef CAFFE_BEAM_SEARCH_LAYER_HPP_
#define CAFFE_BEAM_SEARCH_LAYER_HPP_

#include <vector>
#include <queue>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_beam_search_layer.hpp"

namespace caffe {

/**
 * @brief Performs Beam Search decoding of "Long Short-Term Memory" (LSTM)
 *        style recurrent neural networks (RNNs) using the LSTMNode layer implementation.
 *        A single timestep of the RNN should be defined in a proto file and provided as
 *        a parameter, along with the beam size, a list of recurrent connections, etc.
 * @param bottom input Blob vector (length 1+)
 *     -# @f$ (N \times X1 \times X2 \times X3) @f$
 *        the non-recurrent inputs to the RNN (e.g. data, CNN output, etc)
 * @param top output Blob vector (length 3)
 *     -# @f$ (N \times 1 \times B \times T) @f$ 
 *        sequences of output indices of (maximum) length @f$ T @f$, ordered by 
 *        decreasing summed scores, for each beam @f$ B @f$. 
 *     -# @f$ (N \times 1 \times B) @f$ 
 *        the summed score for each beam sequence @f$ B @f$. 
 *     -# @f$ (N \times 1 \times B \times T) @f$ 
 *        output scores of (maximum) length @f$ T @f$ for each beam @f$ B @f$. 
 */
template <typename Dtype>
class BeamSearchLayer : public BaseBeamSearchLayer<Dtype> {
 public:
  explicit BeamSearchLayer(const LayerParameter& param)
      : BaseBeamSearchLayer<Dtype>(param) {}

  virtual inline const char* type() const { return "BeamSearch"; }
  virtual inline int MinBottomBlobs() const { return 1; }

 protected:

  // Struct used by the cpu layer version
  class BeamExpansion{
    public:
      Dtype score;
      vector<int> words;
      vector<Dtype> score_seq;
      int index;

      bool operator<(BeamExpansion other) const {
        return score > other.score; // Reversed to get largest sort
      }

  };

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);


};

}  // namespace caffe

#endif  // CAFFE_BEAM_SEARCH_LAYER_HPP_
