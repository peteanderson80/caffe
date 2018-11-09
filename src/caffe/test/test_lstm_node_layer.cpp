#include <algorithm>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/lstm_node_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#define NUM_CELLS 3
#define BATCH_SIZE 4
#define INPUT_DATA_SIZE 5

using std::min;
using std::max;

namespace caffe {

template <typename TypeParam>
class LSTMNodeLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  LSTMNodeLayerTest()
      : epsilon_(Dtype(1e-5)),
        blob_bottom_(new Blob<Dtype>()),
        blob_bottom2_(new Blob<Dtype>()),
        blob_bottom_cont_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()),
        blob_top2_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    Caffe::set_random_seed(1601);
    blob_bottom_->Reshape(BATCH_SIZE, INPUT_DATA_SIZE, 1, 1);
    blob_bottom2_->Reshape(BATCH_SIZE, NUM_CELLS, 1, 1);
    blob_bottom_cont_->Reshape(BATCH_SIZE, 1, 1, 1);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_bottom2_);
    caffe_set(BATCH_SIZE, Dtype(1), blob_bottom_cont_->mutable_cpu_data());
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom2_);
    blob_bottom_vec_2_.push_back(blob_bottom_);
    blob_bottom_vec_2_.push_back(blob_bottom2_);
    blob_bottom_vec_2_.push_back(blob_bottom_cont_);
    blob_top_vec_.push_back(blob_top_);
    blob_top_vec_.push_back(blob_top2_);
  }
  virtual ~LSTMNodeLayerTest() { delete blob_bottom_; delete blob_bottom2_; delete blob_top_; delete blob_top2_; }
  void ReferenceLSTMForward(const Blob<Dtype>& blob_bottom,
      const LayerParameter& layer_param, Blob<Dtype>* blob_top);

  Dtype epsilon_;
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom2_;
  Blob<Dtype>* const blob_bottom_cont_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top2_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_bottom_vec_2_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

template <typename TypeParam>
void LSTMNodeLayerTest<TypeParam>::ReferenceLSTMForward(
    const Blob<Dtype>& blob_bottom, const LayerParameter& layer_param,
    Blob<Dtype>* blob_top) {
  typedef typename TypeParam::Dtype Dtype;
  blob_top->Reshape(blob_bottom.num(), blob_bottom.channels(),
      blob_bottom.height(), blob_bottom.width());
  Dtype* top_data = blob_top->mutable_cpu_data();
  LSTMParameter lstm_param = layer_param.lstm_param();
}

TYPED_TEST_CASE(LSTMNodeLayerTest, TestDtypesAndDevices);

TYPED_TEST(LSTMNodeLayerTest, TestSetupAcrossChannels) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  LSTMParameter* lstm_param = layer_param.mutable_lstm_param();
  lstm_param->set_num_cells(NUM_CELLS);
  lstm_param->mutable_input_weight_filler()->set_type("gaussian");
  lstm_param->mutable_input_gate_weight_filler()->set_type("gaussian");
  lstm_param->mutable_forget_gate_weight_filler()->set_type("gaussian");
  lstm_param->mutable_output_gate_weight_filler()->set_type("gaussian");
  lstm_param->set_bias_term(false);

  LSTMNodeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), BATCH_SIZE);
  EXPECT_EQ(this->blob_top_->channels(), NUM_CELLS);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);

  EXPECT_EQ(this->blob_top2_->num(), BATCH_SIZE);
  EXPECT_EQ(this->blob_top2_->channels(), NUM_CELLS);
  EXPECT_EQ(this->blob_top2_->height(), 1);
  EXPECT_EQ(this->blob_top2_->width(), 1);
}

TYPED_TEST(LSTMNodeLayerTest, TestGradientAcrossChannels) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  LSTMParameter* lstm_param = layer_param.mutable_lstm_param();
  lstm_param->set_num_cells(NUM_CELLS);
  lstm_param->mutable_input_weight_filler()->set_type("gaussian");
  lstm_param->mutable_input_gate_weight_filler()->set_type("gaussian");
  lstm_param->mutable_forget_gate_weight_filler()->set_type("gaussian");
  lstm_param->mutable_output_gate_weight_filler()->set_type("gaussian");
  lstm_param->set_bias_term(false);

  LSTMNodeLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 1);
}

TYPED_TEST(LSTMNodeLayerTest, TestSetupAcrossChannelsWithBiases) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  LSTMParameter* lstm_param = layer_param.mutable_lstm_param();
  lstm_param->set_num_cells(NUM_CELLS);
  lstm_param->mutable_input_weight_filler()->set_type("gaussian");
  lstm_param->mutable_input_gate_weight_filler()->set_type("gaussian");
  lstm_param->mutable_forget_gate_weight_filler()->set_type("gaussian");
  lstm_param->mutable_output_gate_weight_filler()->set_type("gaussian");
  lstm_param->mutable_input_bias_filler()->set_type("uniform");
  lstm_param->mutable_input_gate_bias_filler()->set_type("uniform");
  lstm_param->mutable_forget_gate_bias_filler()->set_type("uniform");
  lstm_param->mutable_output_gate_bias_filler()->set_type("uniform");

  LSTMNodeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), BATCH_SIZE);
  EXPECT_EQ(this->blob_top_->channels(), NUM_CELLS);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
}

TYPED_TEST(LSTMNodeLayerTest, TestGradientAcrossChannelsWithBiases) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  LSTMParameter* lstm_param = layer_param.mutable_lstm_param();
  lstm_param->set_num_cells(NUM_CELLS);
  lstm_param->mutable_input_weight_filler()->set_type("gaussian");
  lstm_param->mutable_input_gate_weight_filler()->set_type("gaussian");
  lstm_param->mutable_forget_gate_weight_filler()->set_type("gaussian");
  lstm_param->mutable_output_gate_weight_filler()->set_type("gaussian");
  lstm_param->mutable_input_bias_filler()->set_type("uniform");
  lstm_param->mutable_input_bias_filler()->set_min(-2);
  lstm_param->mutable_input_bias_filler()->set_max(2);
  lstm_param->mutable_input_gate_bias_filler()->set_type("uniform");
  lstm_param->mutable_input_gate_bias_filler()->set_min(-2);
  lstm_param->mutable_input_gate_bias_filler()->set_max(2);
  lstm_param->mutable_forget_gate_bias_filler()->set_type("uniform");
  lstm_param->mutable_forget_gate_bias_filler()->set_min(-2);
  lstm_param->mutable_forget_gate_bias_filler()->set_max(2);
  lstm_param->mutable_output_gate_bias_filler()->set_type("uniform");
  lstm_param->mutable_output_gate_bias_filler()->set_min(-2);
  lstm_param->mutable_output_gate_bias_filler()->set_max(2);

  LSTMNodeLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 1);
}

TYPED_TEST(LSTMNodeLayerTest, TestGradientWithContinuationIndicator) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  LSTMParameter* lstm_param = layer_param.mutable_lstm_param();
  lstm_param->set_num_cells(NUM_CELLS);
  lstm_param->mutable_input_weight_filler()->set_type("gaussian");
  lstm_param->mutable_input_gate_weight_filler()->set_type("gaussian");
  lstm_param->mutable_forget_gate_weight_filler()->set_type("gaussian");
  lstm_param->mutable_output_gate_weight_filler()->set_type("gaussian");
  lstm_param->mutable_input_bias_filler()->set_type("uniform");
  lstm_param->mutable_input_bias_filler()->set_min(-2);
  lstm_param->mutable_input_bias_filler()->set_max(2);
  lstm_param->mutable_input_gate_bias_filler()->set_type("uniform");
  lstm_param->mutable_input_gate_bias_filler()->set_min(-2);
  lstm_param->mutable_input_gate_bias_filler()->set_max(2);
  lstm_param->mutable_forget_gate_bias_filler()->set_type("uniform");
  lstm_param->mutable_forget_gate_bias_filler()->set_min(-2);
  lstm_param->mutable_forget_gate_bias_filler()->set_max(2);
  lstm_param->mutable_output_gate_bias_filler()->set_type("uniform");
  lstm_param->mutable_output_gate_bias_filler()->set_min(-2);
  lstm_param->mutable_output_gate_bias_filler()->set_max(2);

  LSTMNodeLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  for (int i=0; i<BATCH_SIZE; i++){
    this->blob_bottom_vec_2_[2]->mutable_cpu_data()[i] = i%2; // 0 or 1 inputs
  }
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_2_,
      this->blob_top_vec_, 0);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_2_,
      this->blob_top_vec_, 1);
  caffe_set(BATCH_SIZE, Dtype(1), this->blob_bottom_vec_2_[2]->mutable_cpu_data());
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_2_,
      this->blob_top_vec_, 0);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_2_,
      this->blob_top_vec_, 1);
}


}  // namespace caffe
