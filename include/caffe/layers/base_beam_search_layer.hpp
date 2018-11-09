#ifndef CAFFE_BASE_BEAM_SEARCH_LAYER_HPP_
#define CAFFE_BASE_BEAM_SEARCH_LAYER_HPP_

#include <vector>
#include <queue>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Abstract base class that factors out the code common to
 *        BeamSearchLayer, ConstrainedBeamSearchLayer, and 
 *        SequenceConstrainedBeamSearchLayer.
 */
template <typename Dtype>
class BaseBeamSearchLayer : public Layer<Dtype> {
 public:
  explicit BaseBeamSearchLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}

  virtual inline int ExactNumTopBlobs() const { return 3; }

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return false;
  }

 protected:

  // Helper functions
  // Copy bottom input into the beam search net, duplicating for size of beam search
  void copy_bottom_inputs_cpu(const vector<Blob<Dtype>*>& bottom);
  // Zero input and recurrent connections for first timestep
  void clear_recurrent_inputs_cpu();
#ifndef CPU_ONLY
  // Copy bottom input into the beam search net, duplicating for size of beam search
  void copy_bottom_inputs_gpu(const vector<Blob<Dtype>*>& bottom);
  // Zero input and recurrent connections for first timestep
  void clear_recurrent_inputs_gpu();
  // Sort the current timestep output_score_ in descending order, putting the best expansions for each
  // existing partial sequence are first in the output_score_.diff blob. Corresponding word indices 
  // are saved in output_score_indices.
  void sort_beam_expansions_gpu();
  // Populate score_ for the partial sequence at the current timestep
  // for the 'beam_size' best expansions of each beam.
  void sum_expansion_scores_gpu(const vector<Blob<Dtype>*>& top, const unsigned int timestep);
  // Sort the current timestep score_ in descending order, putting the best new partial sequences
  // for each batch element first. Corresponding word and idx indices are saved in score_indices.
  void sort_scores_gpu();
  // Generate output blobs using the sorted current timestep score_.
  void generate_output(const vector<Blob<Dtype>*>& top, const unsigned int timestep);
  // Copy back the input and recurrent connections for the next timestep, using score_indices to
  // determine the correct offsets
  void copy_back_recurrent_inputs_gpu(const unsigned int timestep);
#endif

  // Constrained beam search implementations can override to return > 1 as required, ie. how
  // many different sets of beams are required for each image. Equal to the number of states
  // in the finite state machine representing constraints.
  virtual int NumStates(const vector<Blob<Dtype>*>& bottom) { return 1; }

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  shared_ptr<Net<Dtype> > net_; // A Net to implement the BeamSearch functionality
  int sequence_length_; // Maximum sequence length
  int beam_size_; // Number of partial sequences kept at each iteration
  int vocab_size_; // Size of the softmax output
  int num_states_; // How many copies of each beam are needed
  int end_of_sequence_; // Value to indicate end of sequence
  bool prevent_repeats_; // Whether to prevent identical tokens from being repeated
  int allowed_multiple_size_; // Count of allowed_multiple_ (allowing for zero value)
  Blob<Dtype> allowed_multiple_; // Tokens that can appear more than once in an output sequence
  // Recurrent connections that will be copied from output to input of net_
  std::vector<std::pair< shared_ptr<Blob<Dtype> >, shared_ptr<Blob<Dtype> > > > recurrent_connections_;
  // Connections from bottom layer index to blobs in net_
  std::vector<std::pair< unsigned int, shared_ptr<Blob<Dtype> > > > bottom_connections_;

  // Blobs for per-timestep calculations
  shared_ptr<Blob<Dtype> > input_sequence_; // Input to next timestep (batch_size * beam_size * num_states)
  shared_ptr<Blob<Dtype> > output_score_; // Score output, which is sorted into diff for each beam (batch_size * beam_size * num_states, vocab_size)
  Blob<int> output_score_indices_; // Original global index offset associated with sorted output_score_ values (batch_size * beam_size * num_states, vocab_size)
  Blob<Dtype> score_; // Scores summed over existing partial sequences (batch_size, beam_size * beam_size * num_states)
  Blob<int> score_indices_; // Indices of score_ after sorting (batch_size, beam_size * beam_size * num_states)

  bool log_reshape_;
};

}  // namespace caffe

#endif  // CAFFE_BASE_BEAM_SEARCH_LAYER_HPP_
