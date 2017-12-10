#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/beam_search_node_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class BeamSearchNodeLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  BeamSearchNodeLayerTest()
      : blob_bottom_sum_(new Blob<Dtype>()),
        blob_bottom_seq_(new Blob<Dtype>()),
        blob_bottom_score_(new Blob<Dtype>()),
        blob_bottom_mem_(new Blob<Dtype>()),
        blob_top_sum_(new Blob<Dtype>()),
        blob_top_seq_(new Blob<Dtype>()),
        blob_top_input_(new Blob<Dtype>()),
        blob_top_mem_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // Reshape bottom blobs
    int seq_length = 2;
    int vocab_size = 5;
    int hidden_size = 8;
    vector<int> shape(2, 1);
    shape[0] = 48;
    this->blob_bottom_sum_->Reshape(shape);
    shape[1] = seq_length;
    this->blob_bottom_seq_->Reshape(shape);
    shape[1] = vocab_size;
    this->blob_bottom_score_->Reshape(shape);
    shape[1] = hidden_size;
    this->blob_bottom_mem_->Reshape(shape);
    Caffe::set_random_seed(1702);
    // fill the seq values, setting values in range[0,4]
    for (int i=0; i<this->blob_bottom_seq_->count(); ++i){
      this->blob_bottom_seq_->mutable_cpu_data()[i] = caffe_rng_rand() % vocab_size;
      // Don't allow 0 unless it's the last token in sequence,
      // Since 0 will be used as end_of_sequence
      if ((i+1)%seq_length != 0 && this->blob_bottom_seq_->mutable_cpu_data()[i] == 0){
        this->blob_bottom_seq_->mutable_cpu_data()[i] = 1;
      }
    }
    // fill the other values
    FillerParameter filler_param;
    GaussianFiller<Dtype> gfiller(filler_param);
    gfiller.Fill(this->blob_bottom_sum_);
    gfiller.Fill(this->blob_bottom_score_);
    gfiller.Fill(this->blob_bottom_mem_);
    // Fill vectors
    blob_bottom_vec_.push_back(blob_bottom_sum_);
    blob_bottom_vec_.push_back(blob_bottom_seq_);
    blob_bottom_vec_.push_back(blob_bottom_score_);
    blob_bottom_vec_.push_back(blob_bottom_mem_);
    blob_bottom_vec_short_.push_back(blob_bottom_score_);
    blob_bottom_vec_short_.push_back(blob_bottom_mem_);
    blob_top_vec_.push_back(blob_top_sum_);
    blob_top_vec_.push_back(blob_top_seq_);
    blob_top_vec_.push_back(blob_top_input_);
    blob_top_vec_.push_back(blob_top_mem_);
  }

  virtual ~BeamSearchNodeLayerTest() {
    delete blob_bottom_sum_; delete blob_bottom_seq_;
    delete blob_bottom_score_; delete blob_bottom_mem_;
    delete blob_top_sum_; delete blob_top_seq_;
    delete blob_top_input_; delete blob_top_mem_;
  }

  Blob<Dtype>* const blob_bottom_sum_;
  Blob<Dtype>* const blob_bottom_seq_;
  Blob<Dtype>* const blob_bottom_score_;
  Blob<Dtype>* const blob_bottom_mem_;
  Blob<Dtype>* const blob_top_sum_;
  Blob<Dtype>* const blob_top_seq_;
  Blob<Dtype>* const blob_top_input_;
  Blob<Dtype>* const blob_top_mem_;
  vector<Blob<Dtype>*> blob_bottom_vec_short_, blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(BeamSearchNodeLayerTest, TestDtypesAndDevices);

TYPED_TEST(BeamSearchNodeLayerTest, TestSetupStepZero) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_beam_search_param()->set_beam_size(3);
  BeamSearchNodeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_short_, this->blob_top_vec_);
  const int batch_size = this->blob_bottom_score_->shape(0);
  EXPECT_EQ(this->blob_top_sum_->shape(0), batch_size);
  EXPECT_EQ(this->blob_top_sum_->shape(1), 1);
  EXPECT_EQ(this->blob_top_seq_->shape(0), batch_size);
  EXPECT_EQ(this->blob_top_seq_->shape(1), 1);
  EXPECT_EQ(this->blob_top_input_->shape(0), batch_size);
  EXPECT_EQ(this->blob_top_input_->shape(1), 1);
  EXPECT_EQ(this->blob_top_mem_->shape(0), batch_size);
  EXPECT_EQ(this->blob_top_mem_->shape(1), this->blob_bottom_mem_->shape(1));
}

TYPED_TEST(BeamSearchNodeLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_beam_search_param()->set_beam_size(3);
  BeamSearchNodeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  const int batch_size = this->blob_bottom_score_->shape(0);
  EXPECT_EQ(this->blob_top_sum_->shape(0), batch_size);
  EXPECT_EQ(this->blob_top_sum_->shape(1), 1);
  EXPECT_EQ(this->blob_top_seq_->shape(0), batch_size);
  EXPECT_EQ(this->blob_top_seq_->shape(1), this->blob_bottom_seq_->shape(1)+1);
  EXPECT_EQ(this->blob_top_input_->shape(0), batch_size);
  EXPECT_EQ(this->blob_top_input_->shape(1), 1);
  EXPECT_EQ(this->blob_top_mem_->shape(0), batch_size);
  EXPECT_EQ(this->blob_top_mem_->shape(1), this->blob_bottom_mem_->shape(1));
}

TYPED_TEST(BeamSearchNodeLayerTest, TestForwardStepZero) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  const int beam_size = 3;
  const int end_of_sequence = 0;
  layer_param.mutable_beam_search_param()->set_beam_size(beam_size);
  layer_param.mutable_beam_search_param()->set_end_of_sequence(end_of_sequence);
  layer_param.mutable_beam_search_param()->set_prevent_repeats(false);
  BeamSearchNodeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_short_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_short_, this->blob_top_vec_);
  const int batch_size = this->blob_bottom_sum_->shape(0);
  const int vocab_size = this->blob_bottom_score_->shape(1);
  for (int n = 0; n < batch_size / beam_size; ++n) {
    std::vector< std::pair<Dtype, int> > scores; // (score, word index)
    // At step zero, only expand the first (empty) beam, to avoid duplicates
    for (int w = 0; w < vocab_size; ++w) {
      Dtype s = this->blob_bottom_score_->cpu_data()[n*beam_size*vocab_size + w];
      scores.push_back(std::make_pair(s,w));
    }
    std::sort(scores.begin(), scores.end(), std::greater<std::pair<Dtype,int> >());
    for (int b = 0; b < beam_size; ++b) {
      const int ix = n*beam_size+b;
      const int word = this->blob_top_input_->cpu_data()[ix];
      CHECK_EQ(Dtype(word), this->blob_top_seq_->cpu_data()[ix]);
      CHECK_EQ(word, scores.at(b).second);
      const Dtype out_score = this->blob_top_sum_->cpu_data()[ix];
      CHECK_EQ(out_score, scores.at(b).first);
      CHECK_EQ(out_score, this->blob_bottom_score_->cpu_data()[n*beam_size*vocab_size+word]);
    }
  }
}

TYPED_TEST(BeamSearchNodeLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  const int beam_size = 3;
  const int end_of_sequence = 0;
  layer_param.mutable_beam_search_param()->set_beam_size(beam_size);
  layer_param.mutable_beam_search_param()->set_end_of_sequence(end_of_sequence);
  layer_param.mutable_beam_search_param()->set_prevent_repeats(false);
  BeamSearchNodeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const int batch_size = this->blob_bottom_sum_->shape(0);
  const int vocab_size = this->blob_bottom_score_->shape(1);
  const int timestep = this->blob_bottom_seq_->shape(1)+1;
  for (int n = 0; n < batch_size / beam_size; ++n) {
    std::vector<std::pair<Dtype, std::pair<int,int> > > scores; // (score, (word index, source beam index))
    for (int b = 0; b < beam_size; ++b) {
      const int src_ix = (n*beam_size+b)*this->blob_bottom_seq_->shape(1);
      const int prev_word_ix = src_ix +this->blob_bottom_seq_->shape(1)-1;
      Dtype prev_word = this->blob_bottom_seq_->cpu_data()[prev_word_ix];
      if (prev_word == Dtype(end_of_sequence)){
        Dtype s = this->blob_bottom_sum_->cpu_data()[n*beam_size + b];
        scores.push_back(std::make_pair(s, std::make_pair(end_of_sequence,src_ix)));
      } else {
        for (int w = 0; w < vocab_size; ++w) {
          Dtype s = this->blob_bottom_sum_->cpu_data()[n*beam_size + b] +
            this->blob_bottom_score_->cpu_data()[(n*beam_size + b)*vocab_size + w];
          scores.push_back(std::make_pair(s, std::make_pair(w,src_ix)));
        }
      }
    }
    std::sort(scores.begin(), scores.end(), std::greater<std::pair<Dtype, std::pair<int,int> > >());
    for (int b = 0; b < beam_size; ++b) {
      // Check word output
      const int ix = n*beam_size+b;
      const int word = this->blob_top_input_->cpu_data()[ix];
      CHECK_EQ(Dtype(word), this->blob_top_seq_->cpu_data()[ix*timestep+timestep-1]);
      CHECK_EQ(word, scores.at(b).second.first);
      // Check score output
      const Dtype out_score = this->blob_top_sum_->cpu_data()[ix];
      CHECK_EQ(out_score, scores.at(b).first) << "n=" << n << ", b=" << b;
      // Check sequence output
      int num_inputs = this->blob_bottom_seq_->shape(1);
      int src_ix = scores.at(b).second.second;
      for (int i=0; i<num_inputs; ++i){
        CHECK_EQ(this->blob_bottom_seq_->cpu_data()[src_ix+i],
          this->blob_top_seq_->cpu_data()[(n*beam_size+b)*timestep+i])
          << "n=" << n << ", b=" << b << ",i=" << i << ", src_ix=" << src_ix;
      }
      // Check recurrent outputs
      if (word != end_of_sequence) {
        int hidden_size = this->blob_bottom_mem_->shape(1);
        for (int i=0; i<hidden_size; ++i){
          CHECK_EQ(this->blob_bottom_mem_->cpu_data()[src_ix/num_inputs*hidden_size+i],
            this->blob_top_mem_->cpu_data()[(n*beam_size+b)*hidden_size+i])
            << "n=" << n << ", b=" << b << ",i=" << i << ", src_ix=" << src_ix;
        }
      }
    }
  }
}

TYPED_TEST(BeamSearchNodeLayerTest, TestForwardNoRepeats) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  const int beam_size = 3;
  const int end_of_sequence = 0;
  layer_param.mutable_beam_search_param()->set_beam_size(beam_size);
  layer_param.mutable_beam_search_param()->set_end_of_sequence(end_of_sequence);
  layer_param.mutable_beam_search_param()->set_prevent_repeats(true);
  BeamSearchNodeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const int batch_size = this->blob_bottom_sum_->shape(0);
  const int timestep = this->blob_bottom_seq_->shape(1)+1;
  for (int n = 0; n < batch_size / beam_size; ++n) {
    for (int b = 0; b < beam_size; ++b) {
      // Check word output - shortcut test, only checks this
      const int ix = n*beam_size+b;
      const int word = this->blob_top_seq_->cpu_data()[ix*timestep+timestep-1];
      const int prev_word = this->blob_top_seq_->cpu_data()[ix*timestep+timestep-2];
      if (word != end_of_sequence){
        CHECK_NE(Dtype(word), Dtype(prev_word))
          << "n=" << n << ", b=" << b << " ix=" << ix;
      }
    }
  }
}

TYPED_TEST(BeamSearchNodeLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_beam_search_param()->set_beam_size(3);
  layer_param.mutable_beam_search_param()->set_end_of_sequence(0);
  layer_param.mutable_beam_search_param()->set_prevent_repeats(false);
  BeamSearchNodeLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradient(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0); // bottom blob 0 is sum
  checker.CheckGradient(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 2); // bottom blob 2 is score
  checker.CheckGradient(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 3); // bottom blob 3 is recurrent
}

TYPED_TEST(BeamSearchNodeLayerTest, TestGradientNoRepeats) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_beam_search_param()->set_beam_size(3);
  layer_param.mutable_beam_search_param()->set_end_of_sequence(0);
  layer_param.mutable_beam_search_param()->set_prevent_repeats(true);
  BeamSearchNodeLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradient(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0); // bottom blob 0 is sum
  checker.CheckGradient(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 2); // bottom blob 2 is score
  checker.CheckGradient(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 3); // bottom blob 3 is recurrent
}

TYPED_TEST(BeamSearchNodeLayerTest, TestScoreGradientTimeZero) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_beam_search_param()->set_beam_size(3);
  layer_param.mutable_beam_search_param()->set_end_of_sequence(0);
  layer_param.mutable_beam_search_param()->set_prevent_repeats(false);
  BeamSearchNodeLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradient(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 2); // bottom blob 2 is score
}


}  // namespace caffe
