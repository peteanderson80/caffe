#ifndef CAFFE_BEAM_SEARCH_NODE_LAYER_HPP_
#define CAFFE_BEAM_SEARCH_NODE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Performs a singe step of Beam Search.
 */

/**
 * @brief Performs a single step of Beam Search decoding, adding an additional
 *        output token to an input set of partial sequences, while also
 *        generating recurrent network inputs for the next time step. Note that beams 
 *        associated with the same input data must be adjacent in the minibatch, 
 *        which will have dimension @f$ N = batch_size \times beam_size @f$.
 * @param bottom input Blob vector (length 1+)
 *     -# @f$ (N \times 1) @f$
 *        (optional) partial sequence scores.
 *     -# @f$ (N \times S) @f$
 *        (optional) partial sequences of length @f$ S @f$.
 *     -# @f$ (N \times V) @f$
 *        scores for each of the @f$ V @f$ possible next outputs.
 *     -# @f$ (N \times C) @f$
 *        (optional) additional recurrent connections.
 * @param top output Blob vector (length 3+)
 *     -# @f$ (N \times 1) @f$
 *        updated partial sequence scores.
 *     -# @f$ (N \ times (S+1)) @f$
 *        updated partial sequences of length @f$ S+1 @f$.
 *     -# @f$ (N \times 1) @f$
 *        network inputs for the next time step.
 *     -# @f$ (N \times C) @f$
 *        (optional) recurrent connection inputs for the next time step. 
 */

template <typename Dtype>
class BeamSearchNodeLayer : public Layer<Dtype> {
 public:
  explicit BeamSearchNodeLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "BeamSearchNode"; }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 3; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual int NonRecurrentOutputs() { return 3; }
#ifndef CPU_ONLY
  // Fast concurrent reverse sorting of multiple arrays. Each chunksize of data 
  // in score and index is sorted as a separate array using 'back-to-back' sorting,
  // based on score data. Makes temporary use of index.diff().
  void reverse_gpu_sort(Blob<Dtype>& score, Blob<int>& index, const int chunksize);
  void recurrent_forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
#endif

  int beam_size_;
  int batch_size_;
  int vocab_size_;
  int timestep_;
  Dtype end_of_sequence_;
  Dtype ignore_label_;
  bool has_input_scores_;
  bool prevent_repeats_; // Whether to prevent identical tokens from being repeated
  int allowed_multiple_size_; // Count of allowed_multiple_ (allowing for zero value)
  Blob<Dtype> allowed_multiple_; // Tokens that can appear more than once in an output sequence
  Blob<int> source_map_; // Record the source index for sequence in the beam
  vector<pair<Dtype,int> > cpu_beams_; // CPU beam sorting
  Blob<Dtype> exp_; // GPU beam sorting - sorted beam expansion scores (N * B, vocab_size)
  Blob<int> exp_indices_; // GPU beam sorting - global index offset associated with exp_ (N * B, vocab_size)
  Blob<Dtype> score_; // GPU beam sorting - scores summed over current partial sequences (N, B * B)
  Blob<int> score_indices_; // GPU beam sorting - indices of score_ after sorting (N, B * B)
};

}  // namespace caffe

#endif  // CAFFE_BEAM_SEARCH_NODE_LAYER_HPP_
