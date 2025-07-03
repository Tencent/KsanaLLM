/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <stdlib.h>
#include <filesystem>

#include "ksana_llm/cache_manager/prefix_cache_manager.h"
#include "ksana_llm/models/baichuan/baichuan_model.h"
#include "ksana_llm/models/gpt/gpt_model.h"
#include "ksana_llm/models/internlm2/internlm_model.h"
#include "ksana_llm/models/internlmxcomposer2/internlmxcomposer2_model.h"
#include "ksana_llm/models/llama/llama_model.h"
#include "ksana_llm/models/llama4/llama4_model.h"
#include "ksana_llm/models/mixtral/mixtral_model.h"
#include "ksana_llm/models/qwen/qwen_model.h"
#include "ksana_llm/models/qwen2_moe/qwen2_moe_model.h"
#include "ksana_llm/models/qwen3_moe/qwen3_moe_model.h"
#include "ksana_llm/samplers/sampler.h"
#include "ksana_llm/utils/calc_intvec_hash.h"
#include "ksana_llm/utils/get_custom_weight_name.h"
#include "ksana_llm/utils/memory_allocator.h"
#include "ksana_llm/utils/search_path.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/string_utils.h"
#include "tests/test.h"

#include "ksana_llm/models/base/fake_weight_for_test.h"
#include "ksana_llm/models/common/model_test_helper.h"
#include "ksana_llm/utils/utils.h"

using namespace ksana_llm;

std::map<std::string, std::vector<std::vector<float>>> model_hidden_stat_baselines;
const char LLAMA_MODEL_NAME[] = "llama";
const char LLAMA4_MODEL_NAME[] = "llama4";
const char BAICHUAN_MODEL_NAME[] = "baichuan";
const char QWEN_MODEL_NAME[] = "qwen";
const char QWEN2VL_MODEL_NAME[] = "qwen2vl";
const char QWEN2MOE_MODEL_NAME[] = "qwen2moe";
const char MIXTRAL_MODEL_NAME[] = "mixtral";
const char QWEN3_MOE_MODEL_NAME[] = "qwen3_moe";
const char INTERNLM2_MODEL_NAME[] = "internlm2";
const char INTERNLMXCOMPOSER_MODEL_NAME[] = "internlmxcomposer2";
const char GPT_MODEL_NAME[] = "gpt";

class TestThresholds {
 public:
  TestThresholds(float prefill_dist_thresh, float decode_dist_thresh, float reqs_dist_thresh, float prefill_perf_thresh,
                 float decode_perf_thresh)
      : prefill_dist_thresh_(prefill_dist_thresh),
        decode_dist_thresh_(decode_dist_thresh),
        reqs_dist_thresh_(reqs_dist_thresh),
        prefill_perf_thresh_(prefill_perf_thresh),
        decode_perf_thresh_(decode_perf_thresh) {}

  float GetPrefillDistThresh() const { return prefill_dist_thresh_; }
  float GetDecodeDistThresh() const { return decode_dist_thresh_; }
  float GetReqsDistThresh() const { return reqs_dist_thresh_; }
  float GetPrefillPerfThresh() const { return prefill_perf_thresh_; }
  float GetDecodePerfThresh() const { return decode_perf_thresh_; }

  void SetPrefillDistThresh(float value) { prefill_dist_thresh_ = value; }
  void SetDecodeDistThresh(float value) { decode_dist_thresh_ = value; }
  void SetReqsDistThresh(float value) { reqs_dist_thresh_ = value; }
  void SetPrefillPerfThresh(float value) { prefill_perf_thresh_ = value; }
  void SetDecodePerfThresh(float value) { decode_perf_thresh_ = value; }

 private:
  float prefill_dist_thresh_;  // Threshold for prefill distance comparison
  float decode_dist_thresh_;   // Threshold for decode distance comparison
  float reqs_dist_thresh_;     // Distance between single req and multi reqs
  float prefill_perf_thresh_;  // Threshold for prefill performance (milliseconds)
  float decode_perf_thresh_;   // Threshold for decode performance (milliseconds)
};

// Define a test configuration class to control the test behavior
struct ModelTestConfig {
  // Flags for which data types to test
  bool test_fp16 = true;
  bool test_bf16 = true;
  bool test_fp8 = false;

  // Flags for which device to test
  bool test_acl = false;

  // Model parameters
  bool add_qkv_bias = false;
  bool use_qk_norm = false;
  bool use_shared_moe = false;
};

void InitModelOutputBaseline();
class FakeTinyWeightTest : public testing::Test {
 protected:
  explicit FakeTinyWeightTest(const std::string &model_config_dir = "tiny_model_configs")
      : model_config_dir_name_(model_config_dir) {}

  std::string model_config_dir_name_;

  void SetUp() override {
    context_ = std::make_shared<Context>(1, 1, 1);
    // 解析 config.json,初始化 ModelConfig 以及 BlockManager
    std::filesystem::path current_path = __FILE__;
    std::filesystem::path parent_path = current_path.parent_path();
    std::filesystem::path config_path_relate = parent_path / "../../../../examples/llama7b/ksana_llm.yaml";
    std::string config_path = std::filesystem::absolute(config_path_relate).string();

    std::filesystem::path model_config_path_relate = parent_path / "../../../../tests/tiny_model_configs";
    std::string model_config_path = std::filesystem::absolute(model_config_path_relate).string();

    const auto &env = Singleton<Environment>::GetInstance();
    env->ParseConfig(config_path);
    env->ParseModelConfig(model_config_path_relate, model_config_path_relate, model_config_dir_name_);
    STATUS_CHECK_FAILURE(env->GetModelConfig("", model_config));

    BlockManagerConfig block_manager_config;
    env->InitializeBlockManagerConfig();
    env->GetBlockManagerConfig(block_manager_config);
    KLLM_LOG_INFO << fmt::format("block_size {}", block_manager_config.device_allocator_config.block_size);
    block_manager_config.device_allocator_config.blocks_num = 10;  // This test just need a few blocks;
    block_manager_config.host_allocator_config.blocks_num = block_manager_config.device_allocator_config.blocks_num;
    KLLM_LOG_INFO << "BlockManager Init finished";

    // Pipeline config is set by InitializeBlockManagerConfig()
    env->GetPipelineConfig(pipeline_config_);

    BlockAllocatorGroupConfig group_1_config;
    group_1_config.devices = {0};
    group_1_config.device_block_num = block_manager_config.device_allocator_config.blocks_num;
    group_1_config.host_block_num = block_manager_config.host_allocator_config.blocks_num;
    group_1_config.block_size = block_manager_config.device_allocator_config.block_size;

    BlockAllocatorManagerConfig block_allocator_manager_config;
    block_allocator_manager_config[1] = group_1_config;

    CacheManagerConfig cache_manager_config;
    cache_manager_config.block_token_num = block_manager_config.device_allocator_config.block_token_num;
    cache_manager_config.tensor_para_size = 1;
    cache_manager_config.swap_threadpool_size = 2;
    cache_manager_config.enable_prefix_caching = true;

    memory_allocator_ = std::make_shared<MemoryAllocator>();
    BlockAllocatorManager block_allocator_manager(block_allocator_manager_config, memory_allocator_, context_,
                                                  block_allocator_creation_fn_);
    block_allocator_group_ = block_allocator_manager.GetBlockAllocatorGroup(1);
    cache_manager_ = std::make_shared<PrefixCacheManager>(cache_manager_config, block_allocator_group_);
  }

  void TearDown() override {}

  std::vector<float> GetLastHiddenState(const std::vector<float> &src, size_t len) {
    if (src.empty()) {
      throw std::invalid_argument("Source vector is empty");
    }
    if (len > src.size()) {
      throw std::out_of_range("Requested length exceeds vector size");
    }
    return {src.end() - len, src.end()};
  }

 protected:
  ModelConfig model_config;
  PipelineConfig pipeline_config_;

  int rank_ = 0;

  std::shared_ptr<Context> context_{nullptr};
  std::shared_ptr<MemoryAllocatorInterface> memory_allocator_ = nullptr;
  BlockAllocatorCreationFunc block_allocator_creation_fn_ = nullptr;
  std::shared_ptr<BlockAllocatorGroupInterface> block_allocator_group_ = nullptr;
  std::shared_ptr<PrefixCacheManager> cache_manager_ = nullptr;

  /*
   * This function tests forward funtion works with fake weight.
   * Other models can be tested by replace ModelType, weight_data_type and Weight
   */
  template <template <class> class ModelType, typename weight_data_type>
  void TestModelInferfaceForward(std::shared_ptr<BaseWeight> llama_weight,
                                 const std::vector<float> &prefill_hidden_state_baseline,
                                 const std::vector<float> &decode_hidden_state_baseline,
                                 const TestThresholds &thresholds) {
    SetDevice(rank_);
#ifdef ENABLE_FP8
    if (model_config.quant_config.method == QUANT_FP8_E4M3 && !context_->IsGemmFp8Supported()) {
      std::cout
          << "model_config.quant_config.method == QUANT_FP8_E4M3 but Cublas is insufficient to support FP8, skip test."
          << std::endl;
      return;
    }
#endif
    KLLM_LOG_DEBUG << "Start ModelType: " << typeid(ModelType<weight_data_type>).name()
                   << ", Type: " << GetTypeString(model_config.weight_data_type) << std::endl;
    std::string log_prefix = fmt::format("[Test: {} ] ", GetTypeString(model_config.weight_data_type));

    Event start;
    Event stop;
    float milliseconds = 0;
    int rounds = 1;
    EventCreate(&start);
    EventCreate(&stop);

    std::shared_ptr<ModelInterface<weight_data_type>> llama = std::make_shared<ModelType<weight_data_type>>();

    bool reuse_prefix_config = false;
    std::shared_ptr<FakeModel<weight_data_type>> fake_model = std::make_shared<FakeModel<weight_data_type>>(
        llama, context_, rank_, model_config, pipeline_config_, llama_weight, reuse_prefix_config);

    // ContextDecode
    int hidden_state_len = model_config.head_num * model_config.size_per_head;
    ForwardRequest forward;
    std::vector<int> input_ids = {233, 1681};
    ForwardRequestBuilderForTest request_builder(model_config, cache_manager_);
    request_builder.CreateForwardRequest(1, forward, input_ids);

    std::vector<ForwardRequest> forward_reqs = {forward};
    EXPECT_TRUE(fake_model->Forward(forward_reqs).OK());
    std::vector<float> prefill_output_data;
    fake_model->GetOutputToCPU(prefill_output_data);
    EXPECT_EQ(prefill_output_data.size(), input_ids.size() * hidden_state_len);
    auto prefill_hidden_stat = GetLastHiddenState(prefill_output_data, hidden_state_len);
    float dist = CalculateCosineDist(prefill_hidden_stat, prefill_hidden_state_baseline);
    EXPECT_LT(std::abs(dist), thresholds.GetPrefillDistThresh());
    KLLM_LOG_DEBUG << log_prefix << "last_hidden_state.size=" << hidden_state_len << ", dist to baseline=" << dist
                   << ", data= \n"
                   << Vector2Str(prefill_hidden_stat);

    std::vector<ForwardRequest> multi_forward_reqs = {forward, forward};
    EventRecord(start, context_->GetComputeStreams()[rank_]);
    for (int i = 0; i < rounds; ++i) {
      fake_model->Forward(multi_forward_reqs);
    }

    EventRecord(stop, context_->GetComputeStreams()[rank_]);
    EventSynchronize(stop);
    EventElapsedTime(&milliseconds, start, stop);
    KLLM_LOG_DEBUG << log_prefix << "ContextDecode milliseconds / " << rounds << " is: " << milliseconds / rounds
                   << ", prefill_perf_thresh: " << thresholds.GetPrefillPerfThresh();
    EXPECT_LT((milliseconds / rounds), thresholds.GetPrefillPerfThresh());

    std::vector<float> multi_req_output_data;
    fake_model->GetOutputToCPU(multi_req_output_data);
    auto multi_req_hidden_stat = GetLastHiddenState(prefill_output_data, hidden_state_len);
    dist = CalculateCosineDist(prefill_hidden_stat, multi_req_hidden_stat);
    EXPECT_LT(std::abs(dist), thresholds.GetReqsDistThresh());

    // Only test result is in range [0, vocab_size)
    std::vector<int> sampling_result_tokens = {1};
    EXPECT_LE(0, sampling_result_tokens[0]);
    EXPECT_GT(model_config.vocab_size, sampling_result_tokens[0]);
    (*forward_reqs[0].forwarding_tokens).push_back(sampling_result_tokens[0]);
    sampling_result_tokens.clear();
    for (auto &forward_req : forward_reqs) {
      forward_req.infer_stage = InferStage::STATE_DECODE;
      forward_req.kv_cached_token_num = forward_req.forwarding_tokens->size() - 1;
    }
    size_t forwarding_token_num = forward_reqs.size();
    // Decode
    EXPECT_TRUE(fake_model->Forward(forward_reqs).OK());
    std::vector<float> decode_output_data;
    fake_model->GetOutputToCPU(decode_output_data);
    EXPECT_EQ(decode_output_data.size(), forwarding_token_num * hidden_state_len);
    auto decode_hidden_stat = GetLastHiddenState(decode_output_data, hidden_state_len);
    dist = CalculateCosineDist(decode_hidden_stat, decode_hidden_state_baseline);
    EXPECT_LT(std::abs(dist), thresholds.GetDecodeDistThresh());
    KLLM_LOG_DEBUG << log_prefix << "decode_hidden_stat.size=" << decode_hidden_stat.size()
                   << ", dist to baseline=" << dist << ", data= \n"
                   << Vector2Str(decode_hidden_stat);

    EventRecord(start, context_->GetComputeStreams()[rank_]);
    for (int i = 0; i < rounds; ++i) {
      fake_model->Forward(multi_forward_reqs);
    }
    EventRecord(stop, context_->GetComputeStreams()[rank_]);
    EventSynchronize(stop);
    EventElapsedTime(&milliseconds, start, stop);
    KLLM_LOG_DEBUG << log_prefix << "Decode milliseconds / " << rounds << " is: " << milliseconds / rounds
                   << ", decode_perf_thresh: " << thresholds.GetDecodePerfThresh();
    EXPECT_LT((milliseconds / rounds), thresholds.GetDecodePerfThresh());

    fake_model->GetOutputToCPU(multi_req_output_data);
    multi_req_hidden_stat = GetLastHiddenState(multi_req_output_data, hidden_state_len);
    dist = CalculateCosineDist(decode_hidden_stat, multi_req_hidden_stat);
    EXPECT_LT(std::abs(dist), thresholds.GetReqsDistThresh());
    KLLM_LOG_DEBUG << log_prefix << "multi decode req dist=" << dist;

    llama.reset();
    llama_weight.reset();

    StreamSynchronize(context_->GetMemoryManageStreams()[rank_]);
    EventDestroy(stop);
    EventDestroy(start);
    DeviceSynchronize();
    KLLM_LOG_DEBUG << "Finish ModelType: " << typeid(ModelType<weight_data_type>).name()
                   << ", Type: " << GetTypeString(model_config.weight_data_type) << std::endl;
  }

  /**
   * Tests for different models and weights
   */
  template <template <class> class ModelType, typename weight_data_type, typename WeightType>
  void DoLayerTest(const ModelTestConfig &model_test_config, const std::vector<float> &prefill_hidden_state_baseline,
                   const std::vector<float> &decode_hidden_state_baseline, const TestThresholds &thresholds) {
    DefaultWeightValueInitializer default_weight_initializer;
    std::shared_ptr<BaseWeight> llama_weight =
        std::make_shared<WeightType>(model_config, rank_, model_test_config.add_qkv_bias,
        model_test_config.use_shared_moe, model_test_config.use_qk_norm, &default_weight_initializer);
    TestModelInferfaceForward<ModelType, weight_data_type>(llama_weight, prefill_hidden_state_baseline,
                                                           decode_hidden_state_baseline, thresholds);
  }

  /**
   * Tests for Gpt model specifically
   * Gpt model produces overflow in fp16/fp8 in prefill flash-attention output
   * It causes result to be all Nan
   * So we test Gpt model separately and only assert on the prefill output result as for now
   */
  template <template <class> class ModelType, typename WeightType>
  void DoGptForwardTest(const ModelTestConfig &model_test_config) {
#ifdef ENABLE_ACL
    GTEST_SKIP_("ACL not support this test temporary.");
#endif
    model_config.is_quant = false;
    model_config.weight_data_type = TYPE_FP16;
    model_config.quant_config.method = QUANT_NONE;
    KLLM_LOG_INFO << "Test TYPE_FP16 weight_data_type forward.";
    DefaultWeightValueInitializer default_weight_initializer;
    std::shared_ptr<BaseWeight> gpt_weight =
        std::make_shared<WeightType>(model_config, rank_, model_test_config.add_qkv_bias,
                                     model_test_config.use_shared_moe, &default_weight_initializer);
    std::shared_ptr<ModelInterface<float16>> gpt = std::make_shared<Gpt<float16>>();
    std::shared_ptr<FakeModel<float16>> fake_model =
        std::make_shared<FakeModel<float16>>(gpt, context_, rank_, model_config, pipeline_config_, gpt_weight, false);
    // ContextDecode
    ForwardRequest forward;
    std::vector<int> input_ids = {233, 1681};
    ForwardRequestBuilderForTest request_builder(model_config, cache_manager_);
    request_builder.CreateForwardRequest(1, forward, input_ids);
    std::vector<ForwardRequest> forward_reqs = {forward};
    EXPECT_TRUE(fake_model->Forward(forward_reqs).OK());
    std::vector<float> prefill_output_data;
    fake_model->GetOutputToCPU(prefill_output_data);
    int hidden_state_len = model_config.head_num * model_config.size_per_head;
    EXPECT_EQ(prefill_output_data.size(), input_ids.size() * hidden_state_len);
  }

  template <template <class> class ModelType, typename WeightType>
  void DoForwardTest(const ModelTestConfig &model_test_config, const std::vector<float> &prefill_hidden_state_baseline,
                     const std::vector<float> &decode_hidden_state_baseline, const TestThresholds &thresholds) {
#ifdef ENABLE_TOPS
    GTEST_SKIP_("ZiXiao not support this test temporary.");
#endif
#ifdef ENABLE_ACL
    if (!model_test_config.test_acl) {
      GTEST_SKIP_("ACL not support this test temporary.");
    }
#endif
    // fp16 forward
    if (model_test_config.test_fp16) {
      model_config.is_quant = false;
      model_config.weight_data_type = TYPE_FP16;
      model_config.quant_config.method = QUANT_NONE;
      KLLM_LOG_INFO << "Test TYPE_FP16 weight_data_type forward.";
      DoLayerTest<ModelType, float16, WeightType>(model_test_config, prefill_hidden_state_baseline,
                                                  decode_hidden_state_baseline, thresholds);
    }
#ifdef ENABLE_FP8
    // fp8 forward
    if (model_test_config.test_fp8) {
      model_config.is_quant = true;
      model_config.quant_config.method = QUANT_FP8_E4M3;
      model_config.quant_config.is_checkpoint_fp8_serialized = false;
      KLLM_LOG_INFO << "Test TYPE_FP16 weight_data_type with QUANT_FP8_E4M3 forward";
      DoLayerTest<ModelType, float16, WeightType>(model_test_config, prefill_hidden_state_baseline,
                                                  decode_hidden_state_baseline, thresholds);
    }
#endif

#ifdef ENABLE_CUDA
#  ifdef ENABLE_BFLOAT16
    if (model_test_config.test_bf16) {
      model_config.is_quant = false;
      model_config.weight_data_type = TYPE_BF16;
      model_config.quant_config.method = QUANT_NONE;
      KLLM_LOG_INFO << "Test TYPE_BF16 weight_data_type forward.";
      DoLayerTest<ModelType, bfloat16, WeightType>(model_test_config, prefill_hidden_state_baseline,
                                                   decode_hidden_state_baseline, thresholds);
    }
#    ifdef ENABLE_FP8
    // fp8 forward
    if (model_test_config.test_fp8) {
      model_config.is_quant = true;
      model_config.quant_config.method = QUANT_FP8_E4M3;
      model_config.quant_config.is_checkpoint_fp8_serialized = false;
      KLLM_LOG_INFO << "Test TYPE_BF16 weight_data_type with QUANT_FP8_E4M3 forward";
      DoLayerTest<ModelType, bfloat16, WeightType>(model_test_config, prefill_hidden_state_baseline,
                                                   decode_hidden_state_baseline, thresholds);
    }
#    endif
#  endif
#endif
  }
};

class FakeSimpleModelTest : public FakeTinyWeightTest {
 public:
  FakeSimpleModelTest() : FakeTinyWeightTest("simple_model_config.json") {}
};

class FakeMoeModelTest : public FakeTinyWeightTest {
 public:
  FakeMoeModelTest() : FakeTinyWeightTest("moe_model_config.json") {}
};

class FakeVLModelTest : public FakeTinyWeightTest {
 public:
  FakeVLModelTest() : FakeTinyWeightTest("vl_model_config.json") {}
};

class FakeInternLmXComposerSimpleModelTest : public FakeTinyWeightTest {
 public:
  FakeInternLmXComposerSimpleModelTest() : FakeTinyWeightTest("internlmxcomposer_model_config.json") {}
};

class FakeGptSimpleModelTest : public FakeTinyWeightTest {
 public:
  FakeGptSimpleModelTest() : FakeTinyWeightTest("gpt_model_config.json") {}
};

TEST_F(FakeSimpleModelTest, ForwardTest) {
  InitModelOutputBaseline();
  TestThresholds thresholds(1e-5, 0.004, 0.004, 0.9, 0.9);
#ifdef ENABLE_ACL
  // NPU is slower
  thresholds.SetPrefillPerfThresh(3.8);
  thresholds.SetDecodePerfThresh(3.8);
#endif

  // Simple model test all data types
  ModelTestConfig simple_test_config;
  simple_test_config.test_acl = true;
  simple_test_config.test_fp8 = true;

  simple_test_config.add_qkv_bias = false;
  DoForwardTest<Llama, FakeSimpleWeight>(simple_test_config, model_hidden_stat_baselines[LLAMA_MODEL_NAME][0],
                                         model_hidden_stat_baselines[LLAMA_MODEL_NAME][1], thresholds);

  // Test Baichuan (little different with Llama)
  DoForwardTest<Baichuan, FakeSimpleWeight>(simple_test_config, model_hidden_stat_baselines[BAICHUAN_MODEL_NAME][0],
                                            model_hidden_stat_baselines[BAICHUAN_MODEL_NAME][1], thresholds);

  // Test QWen
  simple_test_config.add_qkv_bias = true;
  DoForwardTest<Qwen, FakeSimpleWeight>(simple_test_config, model_hidden_stat_baselines[QWEN_MODEL_NAME][0],
                                        model_hidden_stat_baselines[QWEN_MODEL_NAME][1], thresholds);

  // Test Internlm
  simple_test_config.add_qkv_bias = false;
  DoForwardTest<Internlm2, FakeSimpleWeight>(simple_test_config, model_hidden_stat_baselines[INTERNLM2_MODEL_NAME][0],
                                             model_hidden_stat_baselines[INTERNLM2_MODEL_NAME][1], thresholds);
}

TEST_F(FakeVLModelTest, ForwardVLTest) {
  InitModelOutputBaseline();
  TestThresholds thresholds(1e-5, 0.004, 0.004, 0.9, 0.9);

  // MultiModal only test fp16
  ModelTestConfig vl_test_config;

  // Test Qwen2VL
  vl_test_config.add_qkv_bias = true;
  DoForwardTest<Qwen, FakeSimpleWeight>(vl_test_config, model_hidden_stat_baselines[QWEN2VL_MODEL_NAME][0],
                                        model_hidden_stat_baselines[QWEN2VL_MODEL_NAME][1], thresholds);
}

TEST_F(FakeMoeModelTest, ForwardMoeTest) {
  InitModelOutputBaseline();
  // Moe model is slower and poor accuracy
  TestThresholds thresholds(1e-4, 0.004, 0.005, 1.9, 1.9);

  // Moe model only test fp16
  ModelTestConfig moe_test_config;

  // Test Qwen2Moe
  moe_test_config.add_qkv_bias = true;
  moe_test_config.use_shared_moe = true;
  DoForwardTest<Qwen2Moe, FakeMoeWeight>(moe_test_config, model_hidden_stat_baselines[QWEN2MOE_MODEL_NAME][0],
                                         model_hidden_stat_baselines[QWEN2MOE_MODEL_NAME][1], thresholds);

  // Test Mixtral after Qwen2_Moe, need moe_config
  moe_test_config.add_qkv_bias = false;
  moe_test_config.use_shared_moe = false;
  DoForwardTest<Mixtral, FakeMoeWeight>(moe_test_config, model_hidden_stat_baselines[MIXTRAL_MODEL_NAME][0],
                                        model_hidden_stat_baselines[MIXTRAL_MODEL_NAME][1], thresholds);

  // Test Qwen3Moe
  moe_test_config.add_qkv_bias = false;
  moe_test_config.use_shared_moe = false;
  moe_test_config.use_qk_norm = true;
  DoForwardTest<Qwen3Moe, FakeMoeWeight>(moe_test_config, model_hidden_stat_baselines[QWEN3_MOE_MODEL_NAME][0],
                                         model_hidden_stat_baselines[QWEN3_MOE_MODEL_NAME][1], thresholds);

  // Test Llama4Moe
  moe_test_config.add_qkv_bias = false;
  moe_test_config.use_shared_moe = true;
  moe_test_config.use_qk_norm = true;
  model_config.moe_config.moe_layers = {0};
  model_config.no_rope_layers = {1, 0};
  DoForwardTest<Llama4, FakeMoeWeight>(moe_test_config, model_hidden_stat_baselines[LLAMA4_MODEL_NAME][0],
                                       model_hidden_stat_baselines[LLAMA4_MODEL_NAME][1], thresholds);
}

TEST_F(FakeInternLmXComposerSimpleModelTest, ForwardTest) {
  InitModelOutputBaseline();
  TestThresholds thresholds(1.2e-4, 0.004, 0.004, 0.9, 0.9);

  // Simple model test all data types
  ModelTestConfig simple_test_config;
  simple_test_config.test_fp8 = true;

  // Test InternlmXComposer2
  simple_test_config.add_qkv_bias = false;
  DoForwardTest<InternlmxComposer2, FakeSimpleWeight>(
      simple_test_config, model_hidden_stat_baselines[INTERNLMXCOMPOSER_MODEL_NAME][0],
      model_hidden_stat_baselines[INTERNLMXCOMPOSER_MODEL_NAME][1], thresholds);
}

TEST_F(FakeGptSimpleModelTest, ForwardGptTest) {
  ModelTestConfig model_test_config;
  DoGptForwardTest<Gpt, FakeGptSimpleWeight>(model_test_config);
}

void InitModelOutputBaseline() {
  std::vector<std::vector<float>> dummy_baseline{2};
  model_hidden_stat_baselines[LLAMA_MODEL_NAME] = dummy_baseline;
  model_hidden_stat_baselines[LLAMA_MODEL_NAME][0] = {
      1.681641f, 1.704102f, 1.700195f, 1.722656f, 1.734375f, 1.727539f, 1.768555f, 1.789062f, 1.758789f, 1.824219f,
      1.783203f, 1.808594f, 1.833984f, 1.810547f, 1.782227f, 1.862305f, 1.910156f, 1.843750f, 1.885742f, 1.872070f,
      1.844727f, 1.866211f, 1.872070f, 1.909180f, 1.934570f, 1.932617f, 1.924805f, 1.942383f, 1.901367f, 1.934570f,
      1.968750f, 2.017578f, 2.007812f, 2.001953f, 2.027344f, 2.017578f, 2.042969f, 2.082031f, 2.085938f, 2.095703f,
      2.089844f, 2.041016f, 2.031250f, 2.080078f, 2.150391f, 2.134766f, 2.154297f, 2.154297f, 2.162109f, 2.197266f,
      2.224609f, 2.201172f, 2.212891f, 2.220703f, 2.236328f, 2.242188f, 2.246094f, 2.251953f, 2.292969f, 2.343750f,
      2.248047f, 2.294922f, 2.306641f, 2.326172f, 1.695312f, 1.742188f, 1.719727f, 1.774414f, 1.739258f, 1.725586f,
      1.759766f, 1.761719f, 1.764648f, 1.791016f, 1.814453f, 1.782227f, 1.820312f, 1.855469f, 1.809570f, 1.827148f,
      1.853516f, 1.854492f, 1.827148f, 1.881836f, 1.911133f, 1.907227f, 1.882812f, 1.943359f, 1.924805f, 1.912109f,
      1.886719f, 1.920898f, 1.925781f, 1.999023f, 1.973633f, 1.984375f, 1.980469f, 2.035156f, 1.977539f, 2.050781f,
      2.048828f, 2.037109f, 2.101562f, 2.103516f, 2.087891f, 2.085938f, 2.087891f, 2.115234f, 2.103516f, 2.128906f,
      2.101562f, 2.103516f, 2.177734f, 2.107422f, 2.171875f, 2.164062f, 2.169922f, 2.259766f, 2.269531f, 2.210938f,
      2.275391f, 2.240234f, 2.261719f, 2.261719f, 2.298828f, 2.302734f, 2.292969f, 2.259766f};
  model_hidden_stat_baselines[LLAMA_MODEL_NAME][1] = {
      0.046753f, -0.021530f, 0.094788f, 0.077698f, 0.088318f, 0.051117f, 0.061157f, 0.085083f, 0.096191f, 0.166138f,
      0.117798f, 0.137207f,  0.179565f, 0.172241f, 0.125732f, 0.177490f, 0.236084f, 0.184937f, 0.183105f, 0.189453f,
      0.146729f, 0.222900f,  0.214233f, 0.198364f, 0.263672f, 0.250977f, 0.250000f, 0.272949f, 0.314209f, 0.269043f,
      0.280762f, 0.369873f,  0.347168f, 0.336914f, 0.327637f, 0.351318f, 0.395020f, 0.439209f, 0.370605f, 0.440918f,
      0.407227f, 0.348145f,  0.364502f, 0.405762f, 0.521973f, 0.484131f, 0.486328f, 0.447021f, 0.506836f, 0.473877f,
      0.534180f, 0.502441f,  0.563477f, 0.542480f, 0.548828f, 0.524902f, 0.546387f, 0.571289f, 0.587402f, 0.629883f,
      0.569824f, 0.641113f,  0.625000f, 0.626465f, 0.015091f, 0.007042f, 0.054535f, 0.085632f, 0.063293f, 0.014755f,
      0.051086f, 0.070068f,  0.125854f, 0.129395f, 0.163574f, 0.115173f, 0.157471f, 0.189087f, 0.163940f, 0.179321f,
      0.177490f, 0.177979f,  0.167969f, 0.175171f, 0.220093f, 0.241211f, 0.205811f, 0.244263f, 0.204834f, 0.228882f,
      0.208862f, 0.269531f,  0.235596f, 0.325439f, 0.306396f, 0.257568f, 0.308350f, 0.362061f, 0.287598f, 0.348389f,
      0.393066f, 0.372314f,  0.385986f, 0.442139f, 0.381592f, 0.415283f, 0.423340f, 0.443848f, 0.418457f, 0.437012f,
      0.426270f, 0.421143f,  0.516113f, 0.464600f, 0.470215f, 0.483643f, 0.529297f, 0.590820f, 0.571289f, 0.536621f,
      0.604004f, 0.582520f,  0.597168f, 0.583008f, 0.596680f, 0.604492f, 0.592773f, 0.572266f};
  model_hidden_stat_baselines[LLAMA4_MODEL_NAME] = dummy_baseline;
  model_hidden_stat_baselines[LLAMA4_MODEL_NAME][0] = {
      -74.25,   201.625,  -153.5,   86.875,   -89.6875, 18.625,   -124.062, 27.5625,  -198.375, 44.1562,  87.375,
      112.375,  -179.125, -90.25,   195.875,  273,      -232.375, 316.25,   -7.17969, -277.5,   253.375,  21.9844,
      -51.0938, 170.625,  390.25,   -12.2812, 118.188,  -312.75,  69.625,   24.125,   284.25,   249.5,    -59.7812,
      34.9688,  -116.312, 436.25,   44.4375,  -343.5,   -414.25,  -329.5,   194.5,    312,      -65.9375, 246.375,
      -354.75,  -150,     178.375,  -337,     133.25,   19.0625,  84.6875,  -299,     -84.625,  15.1953,  535,
      19.5938,  97.625,   34.375,   24.3906,  82.875,   -141.5,   -187,     -12.8125, -29.9219, -110.375, 322.75,
      224.375,  -18.1562, -2.94922, -158,     -17.9219, 220.375,  -4.05078, -319.5,   -190.125, 15.8828,  -72.375,
      -170.625, -403.75,  -22.375,  141.625,  177.75,   165,      -490.5,   105,      -230.375, -285,     -55.375,
      147.125,  -210.875, -293.25,  -489.25,  -68.9375, -532.5,   -33.2188, 110.625,  223.625,  78.25,    137.75,
      435,      215.5,    -231,     73.9375,  -415.75,  -185.25,  -184,     265.5,    344.25,   -34.0312, -255.875,
      -67.4375, -285.75,  -574,     228,      406.5,    -33.2188, 49.6875,  13.1406,  -130.5,   -358.25,  -102.312,
      87.9375,  -70.5625, -37.5625, 99.375,   207.125,  165.375,  324.75};
  model_hidden_stat_baselines[LLAMA4_MODEL_NAME][1] = {
      -143.5,   95.625,   -134.75,  307.25,   -170.625, -74.25,   -185.75,  -112.438, 50.6875,  81.5,     1.60547,
      239.5,    -159.625, 32.0625,  114.438,  396.5,    -132,     144.625,  25.2969,  -173.25,  17.5625,  19.7812,
      -13.4531, 142.5,    335.5,    -18.4219, 107.375,  -164,     137.5,    41.8125,  416,      243.75,   -62.625,
      93.0625,  15.3203,  451.5,    205.25,   -229.125, -198.875, -196.875, 62.0625,  315,      -87.5625, 153.25,
      -494.25,  -17.1719, 207,      -101.688, 192.75,   68.6875,  94.1875,  -238.5,   67.375,   -1.6582,  323.25,
      -90.6875, 20.625,   -108.688, -105.625, 25.3438,  -142.75,  -223.5,   -140.875, -133,     -242.25,  210,
      113.75,   -118.875, -161,     -201.25,  81.6875,  126.938,  174.125,  -409.25,  -84.0625, -73.3125, -287.5,
      -395.5,   -468,     -74.25,   127,      106.812,  76.3125,  -408,     146.625,  -79.5,    -227,     -39.5938,
      319.75,   -137.75,  -293.25,  -479.25,  -302.75,  -441.25,  -170.375, 294.25,   207,      53.6875,  130.875,
      451.75,   202.5,    -262.5,   153.75,   -322.75,  -75.75,   -87.9375, 432.5,    414.75,   -46.75,   -85.1875,
      -74,      -126.875, -265.5,   70.5,     219.125,  -60.5,    3.41406,  -44.9375, 143.875,  -347.75,  -28.8125,
      144.5,    47.2188,  104.625,  78.75,    228.25,   198.5,    209};
  model_hidden_stat_baselines[BAICHUAN_MODEL_NAME] = dummy_baseline;
  model_hidden_stat_baselines[BAICHUAN_MODEL_NAME][0] = {
      1.681641f, 1.704102f, 1.700195f, 1.722656f, 1.734375f, 1.727539f, 1.768555f, 1.789062f, 1.758789f, 1.824219f,
      1.783203f, 1.808594f, 1.833984f, 1.810547f, 1.782227f, 1.862305f, 1.910156f, 1.843750f, 1.885742f, 1.872070f,
      1.844727f, 1.866211f, 1.872070f, 1.909180f, 1.934570f, 1.932617f, 1.924805f, 1.942383f, 1.901367f, 1.934570f,
      1.968750f, 2.017578f, 2.007812f, 2.001953f, 2.027344f, 2.017578f, 2.042969f, 2.082031f, 2.085938f, 2.095703f,
      2.089844f, 2.041016f, 2.031250f, 2.080078f, 2.150391f, 2.134766f, 2.154297f, 2.154297f, 2.162109f, 2.197266f,
      2.224609f, 2.201172f, 2.212891f, 2.220703f, 2.236328f, 2.242188f, 2.246094f, 2.251953f, 2.292969f, 2.343750f,
      2.248047f, 2.294922f, 2.306641f, 2.326172f, 1.695312f, 1.742188f, 1.719727f, 1.774414f, 1.739258f, 1.725586f,
      1.759766f, 1.761719f, 1.764648f, 1.791016f, 1.814453f, 1.782227f, 1.820312f, 1.855469f, 1.809570f, 1.827148f,
      1.853516f, 1.854492f, 1.827148f, 1.881836f, 1.911133f, 1.907227f, 1.882812f, 1.943359f, 1.924805f, 1.912109f,
      1.886719f, 1.920898f, 1.925781f, 1.999023f, 1.973633f, 1.984375f, 1.980469f, 2.035156f, 1.977539f, 2.050781f,
      2.048828f, 2.037109f, 2.101562f, 2.103516f, 2.087891f, 2.085938f, 2.087891f, 2.115234f, 2.103516f, 2.128906f,
      2.101562f, 2.103516f, 2.177734f, 2.107422f, 2.171875f, 2.164062f, 2.169922f, 2.259766f, 2.269531f, 2.210938f,
      2.275391f, 2.240234f, 2.261719f, 2.261719f, 2.298828f, 2.302734f, 2.292969f, 2.259766f};
  model_hidden_stat_baselines[BAICHUAN_MODEL_NAME][1] = {
      0.034302f, -0.058228f, 0.091736f, 0.076477f, 0.095093f, 0.044403f, 0.083069f, 0.073730f, 0.109619f, 0.181763f,
      0.128296f, 0.199707f,  0.152954f, 0.136963f, 0.135132f, 0.178833f, 0.210938f, 0.204956f, 0.195068f, 0.176514f,
      0.157959f, 0.245850f,  0.223877f, 0.201416f, 0.256836f, 0.292725f, 0.265869f, 0.263428f, 0.335693f, 0.272705f,
      0.302734f, 0.390137f,  0.327393f, 0.352295f, 0.347168f, 0.376465f, 0.373779f, 0.456787f, 0.360352f, 0.454102f,
      0.407227f, 0.348633f,  0.380859f, 0.385986f, 0.563965f, 0.469727f, 0.478027f, 0.473389f, 0.530762f, 0.504395f,
      0.505371f, 0.509277f,  0.577148f, 0.576660f, 0.564453f, 0.540527f, 0.581055f, 0.575195f, 0.552246f, 0.611328f,
      0.568848f, 0.643555f,  0.629883f, 0.634277f, 0.012451f, 0.000145f, 0.073303f, 0.106079f, 0.074951f, 0.011124f,
      0.060333f, 0.009415f,  0.123535f, 0.125732f, 0.170654f, 0.089844f, 0.148682f, 0.197632f, 0.198486f, 0.196777f,
      0.189941f, 0.153564f,  0.197632f, 0.193604f, 0.198608f, 0.227051f, 0.227905f, 0.315674f, 0.152100f, 0.232666f,
      0.215332f, 0.298340f,  0.206421f, 0.310059f, 0.279785f, 0.244507f, 0.303467f, 0.372314f, 0.317139f, 0.365234f,
      0.395264f, 0.336914f,  0.401123f, 0.442139f, 0.363037f, 0.402832f, 0.418457f, 0.415527f, 0.398926f, 0.464600f,
      0.428223f, 0.446045f,  0.495605f, 0.452637f, 0.515625f, 0.516602f, 0.536621f, 0.616699f, 0.593750f, 0.510742f,
      0.596680f, 0.569824f,  0.586914f, 0.591797f, 0.617188f, 0.584961f, 0.600098f, 0.604004f};
  model_hidden_stat_baselines[QWEN_MODEL_NAME] = dummy_baseline;
  model_hidden_stat_baselines[QWEN_MODEL_NAME][0] = {
      1.672852f, 1.696289f, 1.703125f, 1.699219f, 1.708008f, 1.725586f, 1.702148f, 1.759766f, 1.793945f, 1.820312f,
      1.809570f, 1.795898f, 1.809570f, 1.838867f, 1.853516f, 1.836914f, 1.844727f, 1.873047f, 1.850586f, 1.861328f,
      1.915039f, 1.896484f, 1.880859f, 1.929688f, 1.999023f, 1.942383f, 1.917969f, 1.917969f, 1.926758f, 1.928711f,
      1.961914f, 1.963867f, 2.042969f, 1.975586f, 2.021484f, 2.074219f, 2.072266f, 2.048828f, 2.035156f, 2.083984f,
      2.037109f, 2.105469f, 2.138672f, 2.080078f, 2.093750f, 2.115234f, 2.173828f, 2.183594f, 2.173828f, 2.193359f,
      2.226562f, 2.179688f, 2.164062f, 2.185547f, 2.195312f, 2.228516f, 2.250000f, 2.210938f, 2.261719f, 2.312500f,
      2.335938f, 2.242188f, 2.289062f, 2.253906f, 1.649414f, 1.659180f, 1.676758f, 1.740234f, 1.737305f, 1.699219f,
      1.757812f, 1.680664f, 1.760742f, 1.764648f, 1.749023f, 1.796875f, 1.822266f, 1.796875f, 1.813477f, 1.785156f,
      1.851562f, 1.875000f, 1.866211f, 1.839844f, 1.842773f, 1.915039f, 1.923828f, 1.908203f, 1.909180f, 1.917969f,
      1.868164f, 1.912109f, 1.974609f, 1.958984f, 1.979492f, 2.039062f, 2.050781f, 1.996094f, 2.046875f, 2.019531f,
      2.046875f, 2.033203f, 2.011719f, 2.117188f, 2.058594f, 2.041016f, 2.126953f, 2.121094f, 2.078125f, 2.072266f,
      2.140625f, 2.187500f, 2.185547f, 2.177734f, 2.195312f, 2.187500f, 2.195312f, 2.214844f, 2.263672f, 2.269531f,
      2.232422f, 2.246094f, 2.242188f, 2.261719f, 2.294922f, 2.291016f, 2.314453f, 2.298828f};
  model_hidden_stat_baselines[QWEN_MODEL_NAME][1] = {
      0.015808f, 0.002613f,  0.014015f, 0.004822f, -0.003716f, 0.038635f,  0.011505f,  0.046967f, 0.051941f, 0.127441f,
      0.122498f, 0.113037f,  0.123169f, 0.161255f, 0.160645f,  0.147949f,  0.134277f,  0.212524f, 0.152100f, 0.183472f,
      0.207520f, 0.207642f,  0.195068f, 0.273193f, 0.297607f,  0.278320f,  0.215576f,  0.249268f, 0.286865f, 0.278320f,
      0.279297f, 0.284668f,  0.352539f, 0.375000f, 0.355713f,  0.395508f,  0.438477f,  0.328613f, 0.328857f, 0.437500f,
      0.353516f, 0.454346f,  0.479736f, 0.419434f, 0.393799f,  0.447021f,  0.484619f,  0.468018f, 0.490479f, 0.514160f,
      0.489990f, 0.485840f,  0.520020f, 0.527344f, 0.588867f,  0.552734f,  0.560059f,  0.550293f, 0.579102f, 0.599609f,
      0.614258f, 0.528809f,  0.612305f, 0.623047f, -0.059967f, -0.033966f, -0.033569f, 0.083191f, 0.042572f, 0.026306f,
      0.065979f, -0.005859f, 0.091553f, 0.087280f, 0.125122f,  0.105103f,  0.138062f,  0.174072f, 0.105591f, 0.088989f,
      0.162720f, 0.186401f,  0.208008f, 0.122070f, 0.160645f,  0.259766f,  0.261963f,  0.218506f, 0.239746f, 0.268066f,
      0.210083f, 0.265869f,  0.280273f, 0.305664f, 0.296631f,  0.345215f,  0.325439f,  0.313232f, 0.387695f, 0.339355f,
      0.394531f, 0.342773f,  0.281250f, 0.438477f, 0.388916f,  0.378174f,  0.432373f,  0.436523f, 0.441650f, 0.366455f,
      0.487305f, 0.484619f,  0.530762f, 0.501465f, 0.506348f,  0.480469f,  0.561035f,  0.550293f, 0.549805f, 0.584473f,
      0.475586f, 0.531250f,  0.608887f, 0.539551f, 0.628418f,  0.613770f,  0.617188f,  0.604980f};
  model_hidden_stat_baselines[QWEN2VL_MODEL_NAME] = dummy_baseline;
  model_hidden_stat_baselines[QWEN2VL_MODEL_NAME][0] = {
      1.672852f, 1.696289f, 1.703125f, 1.699219f, 1.708008f, 1.725586f, 1.702148f, 1.759766f, 1.793945f, 1.820312f,
      1.809570f, 1.795898f, 1.809570f, 1.838867f, 1.853516f, 1.836914f, 1.844727f, 1.873047f, 1.850586f, 1.861328f,
      1.915039f, 1.896484f, 1.880859f, 1.929688f, 1.999023f, 1.942383f, 1.917969f, 1.917969f, 1.926758f, 1.928711f,
      1.961914f, 1.963867f, 2.042969f, 1.975586f, 2.021484f, 2.074219f, 2.072266f, 2.048828f, 2.035156f, 2.083984f,
      2.037109f, 2.105469f, 2.138672f, 2.080078f, 2.093750f, 2.115234f, 2.173828f, 2.183594f, 2.173828f, 2.193359f,
      2.226562f, 2.179688f, 2.164062f, 2.185547f, 2.195312f, 2.228516f, 2.250000f, 2.210938f, 2.261719f, 2.312500f,
      2.335938f, 2.242188f, 2.289062f, 2.253906f, 1.649414f, 1.659180f, 1.676758f, 1.740234f, 1.737305f, 1.699219f,
      1.757812f, 1.680664f, 1.760742f, 1.764648f, 1.749023f, 1.796875f, 1.822266f, 1.796875f, 1.813477f, 1.785156f,
      1.851562f, 1.875000f, 1.866211f, 1.839844f, 1.842773f, 1.915039f, 1.923828f, 1.908203f, 1.909180f, 1.917969f,
      1.868164f, 1.912109f, 1.974609f, 1.958984f, 1.979492f, 2.039062f, 2.050781f, 1.996094f, 2.046875f, 2.019531f,
      2.046875f, 2.033203f, 2.011719f, 2.117188f, 2.058594f, 2.041016f, 2.126953f, 2.121094f, 2.078125f, 2.072266f,
      2.140625f, 2.187500f, 2.185547f, 2.177734f, 2.195312f, 2.187500f, 2.195312f, 2.214844f, 2.263672f, 2.269531f,
      2.232422f, 2.246094f, 2.242188f, 2.261719f, 2.294922f, 2.291016f, 2.314453f, 2.298828f};
  model_hidden_stat_baselines[QWEN2VL_MODEL_NAME][1] = {
      -0.026947f, -0.010742f, 0.028076f,  -0.006466f, 0.011932f, 0.026611f, -0.000603f, 0.049652f, 0.076477f,
      0.140137f,  0.110962f,  0.104980f,  0.129272f,  0.154785f, 0.149658f, 0.153320f,  0.145264f, 0.214722f,
      0.166382f,  0.217407f,  0.178955f,  0.231445f,  0.214111f, 0.257324f, 0.287598f,  0.258545f, 0.238525f,
      0.249268f,  0.256836f,  0.278076f,  0.269775f,  0.248291f, 0.376221f, 0.342041f,  0.356445f, 0.360107f,
      0.430664f,  0.335693f,  0.338867f,  0.402588f,  0.351562f, 0.453369f, 0.493652f,  0.419922f, 0.420166f,
      0.423828f,  0.480469f,  0.494141f,  0.491943f,  0.507324f, 0.475586f, 0.489258f,  0.479248f, 0.566895f,
      0.589844f,  0.533691f,  0.575195f,  0.541016f,  0.560547f, 0.604980f, 0.621094f,  0.533203f, 0.610352f,
      0.608887f,  -0.046997f, -0.020691f, -0.055176f, 0.087341f, 0.029999f, 0.024567f,  0.060760f, -0.013214f,
      0.095215f,  0.085083f,  0.116760f,  0.102905f,  0.132324f, 0.162598f, 0.107910f,  0.113159f, 0.139771f,
      0.179199f,  0.206421f,  0.130493f,  0.161865f,  0.229858f, 0.237061f, 0.227905f,  0.241211f, 0.247681f,
      0.215210f,  0.233521f,  0.298828f,  0.286865f,  0.309082f, 0.357178f, 0.313721f,  0.282715f, 0.416016f,
      0.328369f,  0.406982f,  0.355225f,  0.282959f,  0.399658f, 0.395264f, 0.380859f,  0.454590f, 0.442627f,
      0.450684f,  0.366455f,  0.479248f,  0.477539f,  0.536621f, 0.512207f, 0.523926f,  0.463135f, 0.541504f,
      0.537109f,  0.551758f,  0.572266f,  0.469727f,  0.534180f, 0.603516f, 0.532715f,  0.619629f, 0.621094f,
      0.601562f,  0.632324f};
  model_hidden_stat_baselines[QWEN2MOE_MODEL_NAME] = dummy_baseline;
  model_hidden_stat_baselines[QWEN2MOE_MODEL_NAME][0] = {
      -111.687500f, 156.625000f,  -678.000000f, 390.250000f,  78.125000f,   222.000000f,  214.125000f,  -276.250000f,
      341.500000f,  329.750000f,  -426.000000f, 188.125000f,  -78.500000f,  -48.000000f,  -47.750000f,  -65.750000f,
      489.250000f,  118.687500f,  615.000000f,  -10.125000f,  15.250000f,   -137.750000f, 113.125000f,  705.000000f,
      -222.625000f, 579.000000f,  182.500000f,  54.500000f,   -349.000000f, -78.750000f,  -473.000000f, -407.500000f,
      250.500000f,  226.000000f,  470.500000f,  338.500000f,  208.500000f,  282.000000f,  308.000000f,  -22.000000f,
      163.750000f,  -351.500000f, -241.375000f, -51.406250f,  17.500000f,   -81.375000f,  352.500000f,  -159.375000f,
      218.750000f,  27.937500f,   260.500000f,  209.125000f,  -117.375000f, 120.875000f,  -23.812500f,  -31.375000f,
      84.375000f,   306.000000f,  238.750000f,  32.000000f,   456.500000f,  428.500000f,  161.750000f,  -262.000000f,
      49.625000f,   186.250000f,  294.000000f,  374.000000f,  -271.500000f, -201.375000f, -396.500000f, -266.500000f,
      -174.875000f, 252.750000f,  6.375000f,    -39.375000f,  64.250000f,   85.000000f,   -239.125000f, 187.750000f,
      -355.000000f, 65.250000f,   310.000000f,  19.296875f,   100.187500f,  316.750000f,  -94.375000f,  111.000000f,
      -168.500000f, 180.000000f,  11.843750f,   -493.000000f, 56.875000f,   -790.000000f, -145.375000f, 311.750000f,
      378.000000f,  179.750000f,  -304.000000f, -221.250000f, 32.437500f,   -218.750000f, -191.125000f, 295.000000f,
      -32.125000f,  332.500000f,  -855.500000f, 181.125000f,  -213.500000f, 22.750000f,   -126.437500f, 31.625000f,
      -139.000000f, 700.000000f,  -371.000000f, -135.500000f, 88.875000f,   218.250000f,  -121.250000f, 194.500000f,
      55.937500f,   -245.125000f, 605.000000f,  341.000000f,  -205.250000f, 681.500000f,  1.437500f,    129.250000f};
  model_hidden_stat_baselines[QWEN2MOE_MODEL_NAME][1] = {
      -200.750000f, 23.093750f,   -435.500000f, 212.875000f,  -77.375000f,  2.500000f,    -117.625000f, -529.000000f,
      35.812500f,   311.500000f,  -433.250000f, -32.000000f,  -144.125000f, -46.750000f,  232.375000f,  -361.000000f,
      665.000000f,  129.750000f,  509.000000f,  -32.593750f,  370.750000f,  41.562500f,   268.000000f,  488.250000f,
      -229.500000f, 411.000000f,  -86.312500f,  24.250000f,   186.500000f,  -51.875000f,  -205.500000f, -340.750000f,
      238.000000f,  329.000000f,  618.500000f,  253.500000f,  479.500000f,  241.125000f,  413.000000f,  176.250000f,
      135.625000f,  55.687500f,   -370.000000f, 136.750000f,  -47.656250f,  -190.000000f, 259.500000f,  -222.375000f,
      22.500000f,   -32.937500f,  363.500000f,  359.500000f,  33.750000f,   208.500000f,  55.500000f,   27.984375f,
      -209.250000f, 231.500000f,  -33.218750f,  -49.562500f,  785.500000f,  364.500000f,  236.875000f,  -329.500000f,
      -34.125000f,  244.750000f,  115.625000f,  336.250000f,  -112.250000f, 53.937500f,   -372.750000f, -91.312500f,
      -42.187500f,  229.375000f,  291.250000f,  -51.000000f,  134.250000f,  4.562500f,    -145.000000f, 281.000000f,
      -359.250000f, -113.500000f, 291.000000f,  -81.500000f,  15.218750f,   169.500000f,  -51.750000f,  394.750000f,
      191.500000f,  -98.187500f,  341.500000f,  -207.500000f, -100.875000f, -633.500000f, -52.156250f,  522.000000f,
      180.375000f,  347.500000f,  -273.250000f, -150.750000f, -294.500000f, 45.375000f,   19.750000f,   -159.500000f,
      335.000000f,  260.500000f,  -974.000000f, 222.000000f,  -67.375000f,  348.750000f,  -11.843750f,  -383.500000f,
      85.625000f,   355.750000f,  -273.750000f, -119.375000f, 231.375000f,  461.250000f,  -33.218750f,  229.750000f,
      -45.875000f,  -147.375000f, 283.000000f,  247.750000f,  20.750000f,   929.500000f,  119.750000f,  146.000000f};
  model_hidden_stat_baselines[MIXTRAL_MODEL_NAME] = dummy_baseline;
  model_hidden_stat_baselines[MIXTRAL_MODEL_NAME][0] = {
      1.675781f, 1.675781f, 1.695312f, 1.693359f, 1.723633f, 1.734375f, 1.751953f, 1.736328f, 1.767578f, 1.770508f,
      1.769531f, 1.804688f, 1.813477f, 1.811523f, 1.829102f, 1.822266f, 1.856445f, 1.835938f, 1.877930f, 1.894531f,
      1.863281f, 1.889648f, 1.882812f, 1.901367f, 1.909180f, 1.944336f, 1.937500f, 1.967773f, 1.950195f, 1.950195f,
      1.962891f, 1.989258f, 2.011719f, 2.011719f, 2.033203f, 2.005859f, 2.048828f, 2.044922f, 2.097656f, 2.068359f,
      2.060547f, 2.076172f, 2.099609f, 2.097656f, 2.130859f, 2.130859f, 2.125000f, 2.167969f, 2.152344f, 2.173828f,
      2.195312f, 2.214844f, 2.205078f, 2.199219f, 2.183594f, 2.236328f, 2.236328f, 2.236328f, 2.259766f, 2.261719f,
      2.314453f, 2.289062f, 2.298828f, 2.322266f, 1.686523f, 1.672852f, 1.692383f, 1.717773f, 1.738281f, 1.738281f,
      1.725586f, 1.743164f, 1.745117f, 1.786133f, 1.792969f, 1.795898f, 1.806641f, 1.804688f, 1.837891f, 1.830078f,
      1.849609f, 1.852539f, 1.864258f, 1.901367f, 1.866211f, 1.903320f, 1.918945f, 1.907227f, 1.917969f, 1.946289f,
      1.949219f, 1.969727f, 1.970703f, 2.003906f, 1.988281f, 1.996094f, 1.988281f, 2.013672f, 2.015625f, 2.003906f,
      2.025391f, 2.054688f, 2.054688f, 2.083984f, 2.099609f, 2.103516f, 2.089844f, 2.097656f, 2.119141f, 2.132812f,
      2.160156f, 2.158203f, 2.195312f, 2.154297f, 2.171875f, 2.187500f, 2.195312f, 2.214844f, 2.228516f, 2.250000f,
      2.234375f, 2.251953f, 2.242188f, 2.265625f, 2.281250f, 2.285156f, 2.296875f, 2.292969f};
  model_hidden_stat_baselines[MIXTRAL_MODEL_NAME][1] = {
      0.031677f, -0.035400f, 0.062073f, 0.027832f, 0.054169f, 0.060242f,  0.059601f, 0.022812f, 0.119629f, 0.109741f,
      0.108582f, 0.138916f,  0.094482f, 0.164429f, 0.133667f, 0.117249f,  0.172363f, 0.167358f, 0.189941f, 0.223511f,
      0.188110f, 0.225952f,  0.240845f, 0.212891f, 0.226440f, 0.300293f,  0.274902f, 0.272949f, 0.282715f, 0.294189f,
      0.277588f, 0.319580f,  0.358154f, 0.344482f, 0.321777f, 0.340088f,  0.383545f, 0.395752f, 0.398682f, 0.396729f,
      0.374268f, 0.389648f,  0.408203f, 0.404785f, 0.481201f, 0.465576f,  0.470459f, 0.458252f, 0.473389f, 0.498291f,
      0.501953f, 0.546875f,  0.532227f, 0.545410f, 0.519531f, 0.527344f,  0.582520f, 0.577637f, 0.552246f, 0.553223f,
      0.657715f, 0.604980f,  0.625488f, 0.622070f, 0.014000f, -0.027328f, 0.056885f, 0.061707f, 0.063416f, 0.034180f,
      0.023682f, 0.017059f,  0.063660f, 0.108521f, 0.150757f, 0.094604f,  0.146484f, 0.161255f, 0.194946f, 0.154419f,
      0.190430f, 0.175659f,  0.213379f, 0.232300f, 0.183716f, 0.206177f,  0.248535f, 0.248535f, 0.225830f, 0.269287f,
      0.244873f, 0.294922f,  0.264893f, 0.313232f, 0.290527f, 0.276123f,  0.309570f, 0.327393f, 0.357666f, 0.319336f,
      0.353760f, 0.368408f,  0.360352f, 0.402100f, 0.388916f, 0.424805f,  0.410400f, 0.394287f, 0.422119f, 0.442871f,
      0.457764f, 0.482422f,  0.514648f, 0.477051f, 0.517090f, 0.543457f,  0.543945f, 0.547852f, 0.554199f, 0.554688f,
      0.543945f, 0.563965f,  0.584961f, 0.554199f, 0.579590f, 0.594238f,  0.604492f, 0.620605f};
  model_hidden_stat_baselines[QWEN3_MOE_MODEL_NAME] = dummy_baseline;
  model_hidden_stat_baselines[QWEN3_MOE_MODEL_NAME][0] = {
      3.34375f, 3.375f, 3.39062f, 3.40625f, 3.4375f, 3.45312f, 3.46875f, 3.5f, 3.5f, 3.53125f, 3.5625f, 3.57812f,
      3.59375f, 3.625f, 3.625f, 3.64062f, 3.6875f, 3.70312f, 3.70312f, 3.71875f, 3.75f, 3.76562f, 3.78125f,
      3.82812f, 3.82812f, 3.84375f, 3.875f, 3.89062f, 3.90625f, 3.9375f, 3.95312f, 3.96875f, 4.0f, 4.03125f,
      4.03125f, 4.0625f, 4.0625f, 4.0625f, 4.125f, 4.15625f, 4.15625f, 4.1875f, 4.1875f, 4.1875f, 4.25f,
      4.25f, 4.28125f, 4.3125f, 4.3125f, 4.3125f, 4.375f, 4.375f, 4.40625f, 4.4375f, 4.4375f, 4.4375f, 4.4375f,
      4.5f, 4.53125f, 4.53125f, 4.5625f, 4.5625f, 4.5625f, 4.625f, 3.34375f, 3.375f, 3.39062f, 3.40625f,
      3.4375f, 3.45312f, 3.46875f, 3.5f, 3.5f, 3.53125f, 3.5625f, 3.57812f, 3.59375f, 3.625f, 3.625f,
      3.64062f, 3.6875f, 3.70312f, 3.70312f, 3.71875f, 3.75f, 3.76562f, 3.78125f, 3.82812f, 3.82812f,
      3.84375f, 3.875f, 3.89062f, 3.90625f, 3.9375f, 3.95312f, 3.96875f, 4.0f, 4.03125f, 4.03125f, 4.0625f,
      4.0625f, 4.0625f, 4.125f, 4.15625f, 4.15625f, 4.1875f, 4.1875f, 4.1875f, 4.25f, 4.25f, 4.28125f, 4.3125f,
      4.3125f, 4.3125f, 4.375f, 4.375f, 4.40625f, 4.4375f, 4.4375f, 4.4375f, 4.4375f, 4.5f, 4.53125f, 4.53125f,
      4.5625f, 4.5625f, 4.5625f, 4.625f};
  model_hidden_stat_baselines[QWEN3_MOE_MODEL_NAME][1] = {
      -0.100586f, 0.101074f, 0.00167847f, -0.0168457f, 0.302734f, 0.130859f, 0.416016f, 0.742188f, 0.339844f,
      0.605469f, 0.601562f, 0.929688f, 0.75f, 0.625f, 1.17969f, 1.0f, 1.03125f, 1.375f, 1.07031f, 1.4375f, 1.42188f,
      1.42188f, 1.27344f, 1.48438f, 1.42188f, 1.60156f, 1.82812f, 1.80469f, 1.80469f, 2.03125f, 2.09375f, 2.09375f,
      1.88281f, 2.23438f, 2.21875f, 2.29688f, 2.1875f, 2.375f, 2.29688f, 2.46875f, 2.6875f, 2.84375f, 2.70312f,
      2.84375f, 2.8125f, 2.92188f, 3.0f, 3.09375f, 3.0f, 3.25f, 3.6875f, 3.28125f, 3.5625f, 3.39062f, 3.34375f,
      3.40625f, 3.84375f, 3.75f, 3.8125f, 3.76562f, 4.0625f, 4.1875f, 4.15625f, 4.0625f, 0.0585938f, 0.0375977f,
      0.0395508f, 0.302734f, 0.234375f, 0.388672f, 0.0756836f, 0.488281f, 0.457031f, 0.652344f, 0.757812f,
      0.648438f, 0.976562f, 0.984375f, 0.9375f, 1.13281f, 1.16406f, 1.03906f, 1.15625f, 1.5f, 1.23438f, 1.32812f,
      1.34375f, 1.6875f, 1.46094f, 1.75f, 1.80469f, 1.82812f, 1.61719f, 2.125f, 2.1875f, 2.09375f, 2.28125f,
      2.34375f, 2.26562f, 2.1875f, 2.3125f, 2.35938f, 2.28125f, 2.71875f, 2.5f, 2.5625f, 2.40625f, 2.875f,
      2.75f, 3.03125f, 2.89062f, 3.03125f, 3.25f, 3.34375f, 3.28125f, 3.42188f, 3.23438f, 3.46875f, 3.46875f,
      3.42188f, 3.53125f, 3.65625f, 3.8125f, 3.84375f, 3.8125f, 3.79688f, 3.5625f, 3.98438f};
  model_hidden_stat_baselines[INTERNLM2_MODEL_NAME] = dummy_baseline;
  model_hidden_stat_baselines[INTERNLM2_MODEL_NAME][0] = {
      1.67188f, 1.69531f, 1.70312f, 1.71875f, 1.73438f, 1.72656f, 1.76562f, 1.78906f, 1.75f,    1.82031f, 1.78125f,
      1.80469f, 1.84375f, 1.8125f,  1.78125f, 1.86719f, 1.91406f, 1.84375f, 1.89062f, 1.86719f, 1.85156f, 1.86719f,
      1.86719f, 1.91406f, 1.9375f,  1.92969f, 1.91406f, 1.94531f, 1.89844f, 1.92969f, 1.96094f, 2.01562f, 2.0f,
      2.0f,     2.01562f, 2.03125f, 2.04688f, 2.07812f, 2.07812f, 2.10938f, 2.07812f, 2.0625f,  2.01562f, 2.07812f,
      2.15625f, 2.125f,   2.14062f, 2.15625f, 2.15625f, 2.20312f, 2.23438f, 2.1875f,  2.20312f, 2.23438f, 2.21875f,
      2.25f,    2.25f,    2.26562f, 2.28125f, 2.32812f, 2.25f,    2.29688f, 2.29688f, 2.32812f, 1.69531f, 1.75f,
      1.71875f, 1.77344f, 1.73438f, 1.72656f, 1.76562f, 1.76562f, 1.75781f, 1.79688f, 1.82031f, 1.78125f, 1.82812f,
      1.85938f, 1.80469f, 1.82812f, 1.85156f, 1.85938f, 1.82812f, 1.875f,   1.91406f, 1.90625f, 1.875f,   1.94531f,
      1.92969f, 1.91406f, 1.88281f, 1.92188f, 1.92188f, 1.99219f, 1.97656f, 1.98438f, 1.97656f, 2.03125f, 1.96094f,
      2.03125f, 2.0625f,  2.03125f, 2.09375f, 2.09375f, 2.09375f, 2.09375f, 2.09375f, 2.10938f, 2.10938f, 2.125f,
      2.10938f, 2.10938f, 2.15625f, 2.10938f, 2.17188f, 2.15625f, 2.17188f, 2.26562f, 2.26562f, 2.21875f, 2.28125f,
      2.23438f, 2.26562f, 2.25f,    2.29688f, 2.3125f,  2.28125f, 2.26562f};
  model_hidden_stat_baselines[INTERNLM2_MODEL_NAME][1] = {
      0.0f,      0.00933838f, 0.0153809f, 0.0610352f, 0.0556641f, 0.0544434f, 0.0708008f, 0.0917969f, 0.0947266f,
      0.145508f, 0.0888672f,  0.119629f,  0.164062f,  0.150391f,  0.113281f,  0.178711f,  0.204102f,  0.163086f,
      0.189453f, 0.174805f,   0.147461f,  0.204102f,  0.203125f,  0.212891f,  0.271484f,  0.251953f,  0.253906f,
      0.269531f, 0.279297f,   0.269531f,  0.291016f,  0.328125f,  0.330078f,  0.333984f,  0.357422f,  0.345703f,
      0.384766f, 0.431641f,   0.375f,     0.408203f,  0.404297f,  0.373047f,  0.384766f,  0.412109f,  0.472656f,
      0.488281f, 0.462891f,   0.460938f,  0.488281f,  0.523438f,  0.535156f,  0.535156f,  0.527344f,  0.542969f,
      0.554688f, 0.585938f,   0.558594f,  0.582031f,  0.601562f,  0.664062f,  0.5625f,    0.601562f,  0.632812f,
      0.644531f, 0.0212402f,  0.03125f,   0.0625f,    0.0830078f, 0.0539551f, 0.036377f,  0.0512695f, 0.0893555f,
      0.113281f, 0.11084f,    0.120117f,  0.119629f,  0.141602f,  0.170898f,  0.133789f,  0.154297f,  0.157227f,
      0.207031f, 0.169922f,   0.176758f,  0.21582f,   0.236328f,  0.207031f,  0.228516f,  0.232422f,  0.244141f,
      0.21875f,  0.257812f,   0.235352f,  0.322266f,  0.285156f,  0.283203f,  0.302734f,  0.357422f,  0.291016f,
      0.355469f, 0.384766f,   0.365234f,  0.404297f,  0.429688f,  0.380859f,  0.402344f,  0.398438f,  0.441406f,
      0.453125f, 0.455078f,   0.433594f,  0.412109f,  0.496094f,  0.458984f,  0.488281f,  0.472656f,  0.511719f,
      0.574219f, 0.578125f,   0.554688f,  0.617188f,  0.578125f,  0.570312f,  0.570312f,  0.628906f,  0.628906f,
      0.609375f, 0.585938f};
  model_hidden_stat_baselines[INTERNLMXCOMPOSER_MODEL_NAME] = dummy_baseline;
  model_hidden_stat_baselines[INTERNLMXCOMPOSER_MODEL_NAME][0] = {
      1.67188f, 1.69531f, 1.70312f, 1.71875f, 1.73438f, 1.72656f, 1.76562f, 1.78906f, 1.75f,    1.82031f, 1.78125f,
      1.80469f, 1.84375f, 1.8125f,  1.78125f, 1.86719f, 1.91406f, 1.84375f, 1.89062f, 1.86719f, 1.85156f, 1.86719f,
      1.86719f, 1.91406f, 1.9375f,  1.92969f, 1.91406f, 1.94531f, 1.89844f, 1.92969f, 1.96094f, 2.01562f, 2.0f,
      2.0f,     2.01562f, 2.03125f, 2.04688f, 2.07812f, 2.07812f, 2.10938f, 2.07812f, 2.0625f,  2.01562f, 2.07812f,
      2.15625f, 2.125f,   2.14062f, 2.15625f, 2.15625f, 2.20312f, 2.23438f, 2.1875f,  2.20312f, 2.23438f, 2.21875f,
      2.25f,    2.25f,    2.26562f, 2.28125f, 2.32812f, 2.25f,    2.29688f, 2.29688f, 2.32812f, 1.69531f, 1.75f,
      1.71875f, 1.77344f, 1.73438f, 1.72656f, 1.76562f, 1.76562f, 1.75781f, 1.79688f, 1.82031f, 1.78125f, 1.82812f,
      1.85938f, 1.80469f, 1.82812f, 1.85156f, 1.85938f, 1.82812f, 1.875f,   1.91406f, 1.90625f, 1.875f,   1.94531f,
      1.92969f, 1.91406f, 1.88281f, 1.92188f, 1.92188f, 1.99219f, 1.97656f, 1.98438f, 1.97656f, 2.03125f, 1.96094f,
      2.03125f, 2.0625f,  2.03125f, 2.09375f, 2.09375f, 2.09375f, 2.09375f, 2.09375f, 2.10938f, 2.10938f, 2.125f,
      2.10938f, 2.10938f, 2.15625f, 2.10938f, 2.17188f, 2.15625f, 2.17188f, 2.26562f, 2.26562f, 2.21875f, 2.28125f,
      2.23438f, 2.26562f, 2.25f,    2.29688f, 2.3125f,  2.28125f, 2.26562f};
  model_hidden_stat_baselines[INTERNLMXCOMPOSER_MODEL_NAME][1] = {
      -0.00219727f, 0.00747681f, 0.0127563f, 0.0639648f, 0.0576172f, 0.052002f,  0.0693359f, 0.0922852f, 0.0957031f,
      0.149414f,    0.0859375f,  0.121094f,  0.165039f,  0.150391f,  0.117188f,  0.175781f,  0.207031f,  0.165039f,
      0.191406f,    0.173828f,   0.148438f,  0.206055f,  0.203125f,  0.211914f,  0.271484f,  0.25f,      0.257812f,
      0.269531f,    0.28125f,    0.271484f,  0.287109f,  0.330078f,  0.330078f,  0.335938f,  0.355469f,  0.349609f,
      0.382812f,    0.4375f,     0.373047f,  0.408203f,  0.398438f,  0.375f,     0.386719f,  0.410156f,  0.472656f,
      0.486328f,    0.462891f,   0.460938f,  0.486328f,  0.519531f,  0.53125f,   0.53125f,   0.53125f,   0.546875f,
      0.550781f,    0.582031f,   0.558594f,  0.582031f,  0.601562f,  0.664062f,  0.5625f,    0.601562f,  0.632812f,
      0.644531f,    0.0216064f,  0.0334473f, 0.0620117f, 0.0805664f, 0.0551758f, 0.0371094f, 0.0505371f, 0.0874023f,
      0.114258f,    0.11084f,    0.119629f,  0.121094f,  0.140625f,  0.171875f,  0.131836f,  0.15332f,   0.157227f,
      0.208008f,    0.167969f,   0.173828f,  0.216797f,  0.239258f,  0.206055f,  0.226562f,  0.233398f,  0.242188f,
      0.222656f,    0.257812f,   0.231445f,  0.322266f,  0.287109f,  0.28125f,   0.302734f,  0.359375f,  0.285156f,
      0.353516f,    0.384766f,   0.363281f,  0.402344f,  0.425781f,  0.378906f,  0.402344f,  0.396484f,  0.443359f,
      0.451172f,    0.458984f,   0.433594f,  0.414062f,  0.496094f,  0.455078f,  0.480469f,  0.472656f,  0.511719f,
      0.570312f,    0.578125f,   0.550781f,  0.609375f,  0.574219f,  0.574219f,  0.570312f,  0.628906f,  0.628906f,
      0.609375f,    0.585938f};
}
