/* Copyright 2024 Tencent Inc.  All rights reserved.
 *
 * ==============================================================================*/

#include <cstdlib>

#include "ksana_llm/samplers/sampler.h"
#include "tests/test.h"

using namespace ksana_llm;

class SamplerTest : public testing::Test {
 protected:
  class DerivedSampler : public Sampler {
   public:
    DerivedSampler(const BatchSchedulerConfig &batch_scheduler_config, int rank, std::shared_ptr<Context> context)
        : Sampler(batch_scheduler_config, rank, context) {}

    std::vector<float> GetHostTemperatures() const { return host_temperatures_; }
    std::vector<float> GetNorepeatNgrams() const { return norepeat_ngrams_; }
  };

 protected:
  void SetUp() override {
    context_ = std::make_shared<Context>(1, 1);

    std::filesystem::path current_path = __FILE__;
    std::filesystem::path parent_path = current_path.parent_path();
    std::filesystem::path config_path_relate = parent_path / "../../../examples/llama7b/ksana_llm.yaml";
    std::string config_path = std::filesystem::absolute(config_path_relate).string();

    const auto &env = Singleton<Environment>::GetInstance();
    env->ParseConfig(config_path);
    env->GetModelConfig("", model_config_);
    BlockManagerConfig block_manager_config;
    env->InitializeBlockManagerConfig();
    env->GetBlockManagerConfig(block_manager_config);

    vocab_size_ = model_config_.vocab_size;
    // logits_buf.shape = [max_batch_size, vocab_size]
    // logits_buf.dtype = float32
    Malloc(&logits_buf_, max_batch_size_ * vocab_size_ * sizeof(float));

    BatchSchedulerConfig batch_scheduler_config;
    Singleton<Environment>::GetInstance()->GetBatchSchedulerConfig(batch_scheduler_config);
    sampler_ = std::make_shared<DerivedSampler>(batch_scheduler_config, device_id_, context_);

    // The default sampling mode is greedy.
    sampling_config_.num_beams = 1;
    sampling_config_.topk = 1;
    sampling_config_.topp = 0;
    sampling_config_.temperature = 0;
    sampling_config_.repetition_penalty = 1;
    sampling_config_.no_repeat_ngram_size = 0;
    sampling_config_.encoder_no_repeat_ngram_size = 0;
    sampling_config_.decoder_no_repeat_ngram_size = 0;
    sampling_config_.stop_token_ids = {};
    sampling_config_.max_new_tokens = 1024;
    sampling_config_.logprobs_num = 0;
  }

  void TearDown() override {
    Free(logits_buf_);
    sampler_.reset();
  }

  SamplingRequest GetSamlingRequest() {
    SamplingRequest sample_req;
    sample_req.input_tokens = &token_ids_;
    sample_req.sampling_token_num = 1;
    sample_req.sampling_result_tokens = &sampling_result_tokens_;
    sample_req.forwarding_tokens = &token_ids_;
    sampling_result_tokens_.clear();
    sample_req.logits_offset = 0;
    sample_req.logprobs = &logprobs_;
    sample_req.ngram_dict = &ngram_dict_;
    sample_req.logits_buf = {reinterpret_cast<float *>(logits_buf_)};
    sample_req.model_config = &model_config_;
    sample_req.sampling_config = &sampling_config_;
    return sample_req;
  }

  void SetLogitsBuf(const std::vector<float> &logits_buf_cpu) {
    if (static_cast<int>(logits_buf_cpu.size()) > vocab_size_ * max_batch_size_) {
      KLLM_THROW(fmt::format("logits_buf_cpu is out of space in logits_buf: {} > {}", logits_buf_cpu.size(),
                             vocab_size_ * max_batch_size_));
    }
    MemcpyAsync(logits_buf_, logits_buf_cpu.data(), logits_buf_cpu.size() * sizeof(float), MEMCPY_HOST_TO_DEVICE,
                context_->GetH2DStreams()[device_id_]);
    StreamSynchronize(context_->GetH2DStreams()[device_id_]);
  }

  void SetupNoRepeatInputs() {
    vocab_size_ = 6;
    ngram_size_ = 1;
    output_tokens_ = {1, 2, 3, 4, 5};
    ngram_logits_buf_cpu_.resize(vocab_size_, 0.0f);
    SetLogitsBuf(ngram_logits_buf_cpu_);
  }

  void VerifyNoRepeatNgrams(const std::vector<float> &expected_ngrams) {
    std::vector<float> norepeat_ngrams = sampler_->GetNorepeatNgrams();
    EXPECT_EQ(norepeat_ngrams[5], expected_ngrams[0]);
    EXPECT_EQ(norepeat_ngrams[6], expected_ngrams[1]);
  }

 protected:
  // Parameters used for create the sampler_
  int device_id_ = 0;
  std::shared_ptr<Context> context_{nullptr};
  std::shared_ptr<DerivedSampler> sampler_{nullptr};
  int vocab_size_;
  int max_batch_size_ = 4;

  // Parameters used for default initialization of sample_req.
  void *logits_buf_ = nullptr;
  std::vector<int> token_ids_ = {1, 2, 3, 4, 5};
  std::vector<int> sampling_result_tokens_ = {};
  std::vector<int> draft_tokens_;
  NgramDict ngram_dict_;
  std::vector<std::vector<std::pair<int, float>>> logprobs_;
  ModelConfig model_config_;
  SamplingConfig sampling_config_;

  // Parameters used for no_repeat test
  int ngram_size_;
  std::vector<int> output_tokens_;
  std::vector<float> ngram_logits_buf_cpu_;
};

TEST_F(SamplerTest, DecoderNoRepeatNgramProcessorTest) {
#ifdef ENABLE_TOPS
  GTEST_SKIP_("ZiXiao not support this test temporary.");
#endif
  SetupNoRepeatInputs();
  int input_token_size = output_tokens_.size();

  // Test decoder_no_repeat_ngram_size = 1
  sampler_->DecoderNoRepeatNgramProcessor(reinterpret_cast<float *>(logits_buf_), ngram_size_, input_token_size,
                                          &output_tokens_, &ngram_dict_, vocab_size_,
                                          context_->GetComputeStreams()[device_id_]);
  VerifyNoRepeatNgrams({0, 0});

  output_tokens_.push_back(6);
  sampler_->DecoderNoRepeatNgramProcessor(reinterpret_cast<float *>(logits_buf_), ngram_size_, input_token_size,
                                          &output_tokens_, &ngram_dict_, vocab_size_,
                                          context_->GetComputeStreams()[device_id_]);
  VerifyNoRepeatNgrams({0, -std::numeric_limits<float>::infinity()});
}

TEST_F(SamplerTest, NoRepeatNgramProcessorTest) {
#ifdef ENABLE_TOPS
  GTEST_SKIP_("ZiXiao not support this test temporary.");
#endif
  SetupNoRepeatInputs();
  int input_token_size = output_tokens_.size();

  // Test encoder_no_repeat_ngram_size = 1
  sampler_->EncoderNoRepeatNgramProcessor(reinterpret_cast<float *>(logits_buf_), ngram_size_, input_token_size,
                                          &output_tokens_, &ngram_dict_, vocab_size_,
                                          context_->GetComputeStreams()[device_id_]);
  VerifyNoRepeatNgrams({-std::numeric_limits<float>::infinity(), 0});

  output_tokens_.push_back(6);
  sampler_->EncoderNoRepeatNgramProcessor(reinterpret_cast<float *>(logits_buf_), ngram_size_, input_token_size,
                                          &output_tokens_, &ngram_dict_, vocab_size_,
                                          context_->GetComputeStreams()[device_id_]);
  VerifyNoRepeatNgrams({-std::numeric_limits<float>::infinity(), 0});

  // Test no_repeat_ngram_size = 1
  sampler_->NoRepeatNgramProcessor(reinterpret_cast<float *>(logits_buf_), ngram_size_, input_token_size,
                                   &output_tokens_, &ngram_dict_, vocab_size_,
                                   context_->GetComputeStreams()[device_id_]);
  VerifyNoRepeatNgrams({-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()});
}

TEST_F(SamplerTest, ArgMaxSamplerTest) {
#ifdef ENABLE_TOPS
  GTEST_SKIP_("ZiXiao not support this test temporary.");
#endif
  SamplingRequest sample_req = GetSamlingRequest();

  // Assign a value of 1 to the logits for token_id=6 to make the sampler result as 6.
  std::vector<float> logits_buf_cpu(vocab_size_);
  logits_buf_cpu[6] = 1.f;
  SetLogitsBuf(logits_buf_cpu);
  std::vector<SamplingRequest> sample_reqs = {sample_req};

  sampler_->Sampling(sample_reqs, context_->GetComputeStreams()[device_id_]);
  EXPECT_EQ(1, (*sample_req.sampling_result_tokens).size());
  EXPECT_EQ(6, (*sample_req.sampling_result_tokens).back());
}

TEST_F(SamplerTest, ArgMaxEqualSamplerTest) {
  if (std::getenv("ENABLE_NEW_ARGMAX") == nullptr) {
    return;
  }
#ifdef ENABLE_TOPS
  GTEST_SKIP_("ZiXiao not support this test temporary.");
#endif
  // If there are multiple maximal values then the indices of the first maximal value are returned.
  // Refer to
  // https://pytorch.org/docs/stable/generated/torch.argmax.html
  SamplingRequest sample_req = GetSamlingRequest();

  // Assign a value of 1 to the logits for token_id=3 and token_id=10, and the sampler result
  // should be 3.
  std::vector<float> logits_buf_cpu(vocab_size_);
  logits_buf_cpu[3] = 1.f;
  logits_buf_cpu[10] = 1.f;
  SetLogitsBuf(logits_buf_cpu);
  std::vector<SamplingRequest> sample_reqs = {sample_req};

  sampler_->Sampling(sample_reqs, context_->GetComputeStreams()[device_id_]);
  EXPECT_EQ(1, (*sample_req.sampling_result_tokens).size());
  EXPECT_EQ(3, (*sample_req.sampling_result_tokens).back());
}

TEST_F(SamplerTest, TemperatureAutoVerifyTest) {
#ifdef ENABLE_TOPS
  GTEST_SKIP_("ZiXiao not support this test temporary.");
#endif
  // When the temperature is 0, it should be automatically corrected to 1 to avoid division by zero exception.
  SamplingRequest sample_req = GetSamlingRequest();
  std::vector<SamplingRequest> sample_reqs = {sample_req};

  float *device_logits = nullptr;
  SamplingDeviceParameter sampling_device_parameter;
  sampler_->PrepareDeviceLogitsAndParameter(sample_reqs, sampling_device_parameter, device_logits,
                                            context_->GetComputeStreams()[device_id_]);

  std::vector<float> temperature_cpu = sampler_->GetHostTemperatures();
  EXPECT_NEAR(1.0f, temperature_cpu[0], 1e-6);
}

TEST_F(SamplerTest, LogitsTargetGatherAllTest) {
#ifdef ENABLE_TOPS
  GTEST_SKIP_("ZiXiao not support this test temporary.");
#endif

#ifdef ENABLE_CUDA
  SamplingRequest sample_req = GetSamlingRequest();

  // 设置 request_target 参数，使用 "logits" target 和 GATHER_ALL 模式
  std::map<std::string, ksana_llm::TargetDescribe> request_target;
  ksana_llm::TargetDescribe target_describe;
  target_describe.slice_pos.push_back({0, 1});  // 获取前两个 token 的 logits
  target_describe.token_reduce_mode = TokenReduceMode::GATHER_ALL;
  request_target["logits"] = target_describe;
  sample_req.request_target = &request_target;

  // 设置 logits_custom_length
  sample_req.logits_custom_length = 2;  // 对应 slice_pos 中的 token 数量

  // 初始化 response 成员变量
  std::map<std::string, PythonTensor> response_map;
  sample_req.response = &response_map;

  // 准备 logits 数据
  std::vector<float> logits_buf_cpu(vocab_size_);
  // 设置第一个 token 的 logits
  logits_buf_cpu[6] = 1.0f;  // 第一个 token 的第 6 个 logit 设为 1.0

  SetLogitsBuf(logits_buf_cpu);
  std::vector<SamplingRequest> sample_reqs = {sample_req};

  // 执行 Sampling
  sampler_->Sampling(sample_reqs, context_->GetComputeStreams()[device_id_]);

  // 注意：对于 GATHER_ALL 模式，CopyProbsOutputToRequests 方法会跳过将结果复制到 response 中
  // 因此我们不期望在 response 中找到 "logits" 数据
  // 这是 Sampler 类的预期行为
  EXPECT_TRUE(sample_req.response->find("logits") == sample_req.response->end())
      << "For GATHER_ALL mode, response should not contain logits data";
#endif
}

TEST_F(SamplerTest, LogprobsSamplerTest) {
#ifdef ENABLE_TOPS
  GTEST_SKIP_("ZiXiao not support this test temporary.");
#endif
  SamplingRequest sample_req = GetSamlingRequest();

  std::vector<float> logits_buf_cpu = {0.0, 1.0, 2.0, 1.5, 0.7, 1.8};
  SetLogitsBuf(logits_buf_cpu);

  // set logprobs num to enable logprobs
  sampling_config_.logprobs_num = 2;
  std::vector<SamplingRequest> sample_reqs = {sample_req};

  sampler_->Sampling(sample_reqs, context_->GetComputeStreams()[device_id_]);
  EXPECT_EQ(1, logprobs_.size());

  // logprobs is not supported in ACL.
#ifdef ENABLE_CUDA
  EXPECT_EQ(2, logprobs_[0].size());
  EXPECT_EQ(2, logprobs_[0][0].first);
  EXPECT_NEAR(-0.598139f, logprobs_[0][0].second, 1e-6);
  EXPECT_EQ(5, logprobs_[0][1].first);
  EXPECT_NEAR(-0.798139f, logprobs_[0][1].second, 1e-6);
#endif

  sampling_config_.logprobs_num = 0;
}
