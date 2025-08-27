/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <torch/torch.h>

#include <stdlib.h>
#include <filesystem>

#include "ksana_llm/cache_manager/prefix_cache_manager.h"
#include "ksana_llm/data_hub/data_hub.h"
#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"
#include "ksana_llm/models/base/model_weight.h"
#include "ksana_llm/modules/attention/multihead_latent_attention.h"
#include "ksana_llm/runtime/llm_runtime.h"
#include "ksana_llm/samplers/sampler.h"
#include "ksana_llm/utils/absorb_weights_type.h"
#include "ksana_llm/utils/calc_intvec_hash.h"
#include "ksana_llm/utils/dynamic_memory_pool.h"
#include "ksana_llm/utils/get_custom_weight_name.h"
#include "ksana_llm/utils/memory_allocator.h"
#include "ksana_llm/utils/search_path.h"
#include "ksana_llm/utils/singleton.h"
#include "tests/test.h"

using namespace ksana_llm;

template <typename T>
DataType GetKsanaDataType();
#define GET_KSANA_DATA_TYPE(T, KSANA_TYPE) \
  template <>                              \
  DataType GetKsanaDataType<T>() {         \
    return KSANA_TYPE;                     \
  }
GET_KSANA_DATA_TYPE(int32_t, TYPE_INT32);
GET_KSANA_DATA_TYPE(float, TYPE_FP32);
GET_KSANA_DATA_TYPE(half, TYPE_FP16);
GET_KSANA_DATA_TYPE(__nv_bfloat16, TYPE_BF16);
#undef GET_KSANA_DATA_TYPE

size_t schedule_id = 0;

Status CastDeviceTensorType(Tensor& input_tensor, DataType new_dtype, int dev_rank,
                            std::shared_ptr<Context>& context_) {
#ifdef ENABLE_ACL
  KLLM_THROW(fmt::format("Unsupported tensor cast for Ascend device"));
#elif ENABLE_TOPS
  KLLM_THROW(fmt::format("Unsupported tensor cast for Tops device"));
#endif
  if (input_tensor.dtype == new_dtype) {
    return Status();
  }
#ifdef ENABLE_CUDA
  if (input_tensor.dtype == DataType::TYPE_FP32 && new_dtype == DataType::TYPE_FP16) {
    Tensor new_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, new_dtype, input_tensor.shape, dev_rank, nullptr,
                               &(context_->GetMemoryManageStreams()[dev_rank]));
    CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::FloatToHalf(
        input_tensor.GetPtr<float>(), input_tensor.GetElementNumber(), new_tensor.GetPtr<float16>(),
        context_->GetMemoryManageStreams()[dev_rank].Get()));
    input_tensor = new_tensor;
  } else if (input_tensor.dtype == DataType::TYPE_FP32 && new_dtype == DataType::TYPE_BF16) {
    Tensor new_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, new_dtype, input_tensor.shape, dev_rank, nullptr,
                               &(context_->GetMemoryManageStreams()[dev_rank]));
    CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::FloatToBFloat16(
        input_tensor.GetPtr<float>(), input_tensor.GetElementNumber(), new_tensor.GetPtr<bfloat16>(),
        context_->GetMemoryManageStreams()[dev_rank].Get()));
    input_tensor = new_tensor;
  } else if (input_tensor.dtype == DataType::TYPE_FP16 && new_dtype == DataType::TYPE_FP32) {
    Tensor new_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, new_dtype, input_tensor.shape, dev_rank, nullptr,
                               &(context_->GetMemoryManageStreams()[dev_rank]));
    CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::HalfToFloat(input_tensor.GetPtr<float16>(),
                                                           input_tensor.GetElementNumber(), new_tensor.GetPtr<float>(),
                                                           context_->GetMemoryManageStreams()[dev_rank].Get()));
    input_tensor = new_tensor;
  } else if (input_tensor.dtype == DataType::TYPE_BF16 && new_dtype == DataType::TYPE_FP32) {
    Tensor new_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, new_dtype, input_tensor.shape, dev_rank, nullptr,
                               &(context_->GetMemoryManageStreams()[dev_rank]));
    CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::BFloat16ToFloat(
        input_tensor.GetPtr<bfloat16>(), input_tensor.GetElementNumber(), new_tensor.GetPtr<float>(),
        context_->GetMemoryManageStreams()[dev_rank].Get()));
    input_tensor = new_tensor;
  } else if (input_tensor.dtype == DataType::TYPE_BF16 && new_dtype == DataType::TYPE_FP16) {
    input_tensor.dtype = new_dtype;
    // Inplace cast
    CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::BFloat16ToHalf(input_tensor.GetPtr<void>(),
                                                              input_tensor.GetElementNumber(),
                                                              context_->GetMemoryManageStreams()[dev_rank].Get()));
  } else if (input_tensor.dtype == DataType::TYPE_FP16 && new_dtype == DataType::TYPE_BF16) {
    input_tensor.dtype = new_dtype;
    // Inplace cast
    CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::HalfToBFloat16(input_tensor.GetPtr<void>(),
                                                              input_tensor.GetElementNumber(),
                                                              context_->GetMemoryManageStreams()[dev_rank].Get()));
  } else {
    KLLM_THROW(fmt::format("Unsupported tensor cast from type {} to {}", input_tensor.dtype, new_dtype));
  }
#endif
  return Status();
}

void AssignFromVector(Tensor& tensor, const std::vector<float>& f_vector, std::shared_ptr<Context>& context) {
  DeviceSynchronize();
  int device_rank;
  GetDevice(&device_rank);

  if (f_vector.size() != tensor.GetElementNumber()) {
    KLLM_THROW("Vector size does not match tensor element count");
  }

  Tensor fp32_cast = Tensor(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_FP32, tensor.shape, device_rank, nullptr,
                            &(context->GetMemoryManageStreams()[device_rank]));
  MemcpyAsync(fp32_cast.GetPtr<void>(), f_vector.data(), f_vector.size() * sizeof(float), MEMCPY_HOST_TO_DEVICE,
              context->GetMemoryManageStreams()[device_rank]);
  CastDeviceTensorType(fp32_cast, tensor.dtype, device_rank, context);
  MemcpyAsync(tensor.GetPtr<void>(), fp32_cast.GetPtr<void>(), tensor.GetTotalBytes(), MEMCPY_DEVICE_TO_DEVICE,
              context->GetMemoryManageStreams()[device_rank]);
}

template <typename T>
class MultiHeadLatentAttentionTestModel : public CommonModel {
 public:
  using CommonModel::context_;
  using CommonModel::model_config_;

  ForwardingContext* forwarding_context_;

  using CommonModel::cast_layer_;

  std::shared_ptr<MultiHeadLatentAttention> mla_;
  MlaBuffers mla_buffers_;

  MultiHeadLatentAttentionTestModel(const ModelConfig& model_config, const RuntimeConfig& runtime_config,
                                    const int rank, std::shared_ptr<Context> context,
                                    std::shared_ptr<BaseWeight> base_weight)
      : CommonModel(model_config, runtime_config, rank, context) {
    ModelRunConfig model_run_config;
    model_run_config.position_encoding = PositionEncoding::ROPE;
    CommonModel::InitRunConfig(model_run_config, base_weight);
    CommonModel::AllocResources(schedule_id);
    forwarding_context_ = CommonModel::GetForwardingContext(schedule_id);
  }

  ~MultiHeadLatentAttentionTestModel() { CommonModel::FreeResources(schedule_id); }

  Status CreateLayers(LayerCreationContext& creation_context, ModelCreationConfig& model_creation_config) override {
    MultiHeadLatentAttention::CreateBuffers(CommonModel::GetBufferManager(), model_creation_config.attn_config,
                                            creation_context.runtime_config, mla_buffers_);
    bool is_neox = true;
    int layer_idx = 0;
    mla_ = std::make_shared<MultiHeadLatentAttention>(layer_idx, is_neox, creation_context, model_creation_config,
                                                      mla_buffers_);
    return Status();
  }

  Status LayerForward(ForwardingContext& forwarding_context_, const RunMode run_mode = RunMode::kMain) override {
    return Status();
  }

  Status CommonAttention(const int layer_idx, std::shared_ptr<ksana_llm::BaseWeight>& base_weight,
                         const bool is_multi_token_forward, const std::vector<ForwardRequest>& forward_reqs) {
    forwarding_context_->model_input_->ParseFromRequests(forward_reqs);

    // Set shape and type of hidden unit.
    SetHiddenUnitMeta(forwarding_context_->multi_batch_id_,
                      {forwarding_context_->model_input_->input_ids.shape[0], model_config_.hidden_units},
                      model_config_.weight_data_type);

    // create forward shape tensor
    forwarding_context_->GetAttentionForwardContext().forward_shape.shape = {
        forwarding_context_->model_input_->multi_token_request_num,
        forwarding_context_->model_input_->multi_token_request_max_tokens,
        forwarding_context_->model_input_->flash_input.kv_cache_block_num,
        forwarding_context_->model_input_->single_token_request_num,
        forwarding_context_->model_input_->single_token_request_max_tokens,
        forwarding_context_->model_input_->page_single_input.kv_cache_block_num +
            forwarding_context_->model_input_->page_dual_input.kv_cache_block_num,
        forwarding_context_->model_input_->dp_max_forwarding_tokens,
        forwarding_context_->model_input_->total_prefix_len,
        forwarding_context_->model_input_->dp_multi_token_request_num,
        forwarding_context_->model_input_->dp_multi_token_request_max_tokens,
        forwarding_context_->model_input_->dp_single_token_request_num,
        forwarding_context_->model_input_->dp_single_token_request_max_tokens,
        forwarding_context_->model_input_->dp_total_prefix_len};

    CREATE_BUFFER_SCOPE(hidden_buffer_tensors_0, forwarding_context_->buffers_->hidden_buffer_0);
    CREATE_BUFFER_SCOPE(reduce_buffer_tensors, forwarding_context_->buffers_->shared_buffer);

    forwarding_context_->GetAttentionForwardContext().flag_tensor.template GetPtr<bool>()[0] =
        forwarding_context_->model_input_->use_cache;
    hidden_buffer_tensors_0[0].shape = {forwarding_context_->model_input_->input_ids.shape[0], 2048};
    std::vector<float> input_data(hidden_buffer_tensors_0[0].GetElementNumber());
    for (size_t i = 0; i < input_data.size(); i++) {
      input_data[i] = 1.0f / (i % 97 * 0.1f + 1.0f) * pow(-1, (i % 7));
    }
    AssignFromVector(hidden_buffer_tensors_0[0], input_data, context_);
    Status status = mla_->Forward(hidden_buffer_tensors_0, reduce_buffer_tensors, *forwarding_context_);
    forwarding_context_->GetAttentionForwardContext().forward_shape.shape = {0, 1, 1};
    {
      CREATE_BUFFER_SCOPE(hidden_buffer_tensors_1, forwarding_context_->buffers_->hidden_buffer_1);
      hidden_buffer_tensors_1[0].Fill(0);
      STATUS_CHECK_RETURN(cast_layer_->Forward(
          {hidden_buffer_tensors_0[0], forwarding_context_->GetAttentionForwardContext().forward_shape},
          hidden_buffer_tensors_1));
      Memcpy(input_data.data(), hidden_buffer_tensors_1[0].template GetPtr<void>(),
             sizeof(float) * hidden_buffer_tensors_0[0].GetElementNumber(), MEMCPY_DEVICE_TO_HOST);
    }
    std::vector<float> output;
    if (is_multi_token_forward) {
      output = {-1977, 2088, -3386, 772.5, 125.75, -575.5};
    } else {
      if (GetAbsorbWeightsType() == AbsorbWeightsType::kAbsorbTypeBMM) {
        if (runtime_config_.attn_backend_config.kv_cache_dtype == TYPE_FP8_E4M3) {
          output = {550, 936.5, 365.75, -46.7188, -75.625, -563.5};
        } else {
          output = {559.5, 935.5, 366.25, -55.0312, -76.1875, -564};
        }
      } else {
        output = {595, 399.5, -819.5, 777, -749.5, 1127};
      }
    }

    for (size_t i = 0; i < output.size(); i++) {
      const float diff = std::fabs((input_data[i] - output[i]) / output[i]);
      EXPECT_LT(diff, 0.05);
    }

    return status;
  }
};

template <typename T>
class TestWeight : public BaseWeight {
 public:
  using ksana_llm::BaseWeight::weights_map_;
  TestWeight() {}
  explicit TestWeight(const ksana_llm::ModelConfig& model_config, const ksana_llm::RuntimeConfig& runtime_config,
                      int rank, std::shared_ptr<ksana_llm::Context> context)
      : ksana_llm::BaseWeight(model_config, runtime_config, rank, context) {}
  ~TestWeight() override {}

  Tensor GetModelWeights(const std::string& weight_name) override {
    auto it = weights_map_.find(weight_name);
    if (it == weights_map_.end()) {
      KLLM_LOG_WARNING << fmt::format("weight_name: {} not in weights map", weight_name);
      return Tensor();
    }
    return it->second;
  }

  virtual Status LoadWeightsFromFile(const std::shared_ptr<BaseFileTensorLoader> weights_loader,
                                     const std::vector<std::string>& weight_name_list,
                                     const std::vector<std::string>& custom_name_list) override {
    return Status();
  }

  virtual void ProcessWeights() override { return; }
  virtual void SetEmbeddingsConfig() override { return; }

  void AddMlaWeight(int device_id = 0) {
    std::unordered_map<std::string, std::vector<size_t>> add_tensor_map;
    const int layer_num = 4;
    for (int i = 0; i < layer_num; i++) {
      add_tensor_map[fmt::format("model.layers.{}.self_attn.kv_a_layernorm.weight", i)] = {512};
      add_tensor_map[fmt::format("model.layers.{}.self_attn.o_proj.weight", i)] = {2048, 2048};
      add_tensor_map[fmt::format("model.layers.{}.post_attention_layernorm.weight", i)] = {2048};
      add_tensor_map[fmt::format("model.layers.{}.input_layernorm.weight", i)] = {2048};
      add_tensor_map[fmt::format("model.layers.{}.self_attn.kv_a_rope_proj.weight", i)] = {2048, 64};
      add_tensor_map[fmt::format("model.layers.{}.self_attn.v_head_proj.weight", i)] = {512, 2048};
      add_tensor_map[fmt::format("model.layers.{}.self_attn.kv_a_lora_proj.weight", i)] = {2048, 512};
      add_tensor_map[fmt::format("model.layers.{}.self_attn.kv_b_nope_proj.weight", i)] = {512, 2048};
      add_tensor_map[fmt::format("model.layers.{}.self_attn.w_uk_t.weight", i)] = {16, 128, 512};
      add_tensor_map[fmt::format("model.layers.{}.self_attn.w_uv.weight", i)] = {16, 512, 128};
      add_tensor_map[fmt::format("model.layers.{}.self_attn.q_b_nope_rope_proj.weight", i)] = {2048, 3072};
    }
    DataType weight_type = GetKsanaDataType<T>();
    for (auto& [tensor_name, shape] : add_tensor_map) {
      weights_map_[tensor_name] = Tensor(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_FP32, shape, device_id,
                                         nullptr, &(context_->GetMemoryManageStreams()[device_id]));
      Tensor& tensor = weights_map_.at(tensor_name);
      std::vector<float> input_data(tensor.GetElementNumber());
      for (size_t i = 0; i < input_data.size(); i++) {
        input_data[i] = 1.0f / (i % 97 * 0.1f + 1.0f) * pow(-1, (i % 7));
      }
      MemcpyAsync(tensor.template GetPtr<void>(), reinterpret_cast<void*>(input_data.data()), tensor.GetTotalBytes(),
                  MEMCPY_HOST_TO_DEVICE, context_->GetMemoryManageStreams()[device_id]);
      CastDeviceTensorType(tensor, weight_type, device_id, context_);
    }
  }

  void SaveToNpy() {
    const std::string save_prefix = "Tensors_new/";
    for (auto& [tensor_name, tensor] : weights_map_) {
      tensor.SaveToNpyFile(save_prefix + tensor_name + "_" + std::to_string(static_cast<int>(tensor.dtype)) + ".npy");
    }
  }
};

// 定义一个 MlaTest 类,继承自 testing::Test
class MlaTest : public testing::Test {
 protected:
  void SetUp() override {
    origin_stderr_verbosity = loguru::g_stderr_verbosity;
    loguru::g_stderr_verbosity = loguru::Verbosity_MAX;
    DeviceMemoryPool::Disable();

    const auto* test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    test_name = test_info->name();
    if (test_name.find("ForwardNewAbsorbWithFlashMla") != std::string::npos) {
      SetAbsorbWeightsType(AbsorbWeightsType::kAbsorbTypeBMM);
    }

    context_ = std::make_shared<Context>(1, 1, 1);
    // 解析 config.json,初始化 ModelConfig 以及 BlockManager
    std::filesystem::path current_path = __FILE__;
    std::filesystem::path parent_path = current_path.parent_path();
    std::filesystem::path config_path_relate = parent_path / "../../../../examples/ksana_llm_deepseekv2.yaml";
    std::string config_path = std::filesystem::absolute(config_path_relate).string();

    const auto& env = Singleton<Environment>::GetInstance();
    env->ParseConfig(config_path, std::filesystem::absolute(parent_path / "../../../../examples/deepseekv2/").string());

    env->GetModelConfig(model_config);

    BlockManagerConfig block_manager_config;
    if (test_name.find("ForwardNewAbsorbWithFlashMlaKvFP8Test") != std::string::npos) {
      env->GetBlockManagerConfig(block_manager_config);
      // kv_cache_dtype must be set before calling env->InitializeBlockManagerConfig()
      block_manager_config.host_allocator_config.kv_cache_dtype = DataType::TYPE_FP8_E4M3;
      block_manager_config.device_allocator_config.kv_cache_dtype = DataType::TYPE_FP8_E4M3;
      env->SetBlockManagerConfig(block_manager_config);
      KLLM_LOG_INFO << "Exec Test ForwardNewAbsorbWithFlashMlaKvFP8Test";
    }
    env->InitializeBlockManagerConfig();
    env->GetBlockManagerConfig(block_manager_config);
    block_manager_config.block_host_memory_factor = 0;
    block_manager_config.reserved_device_memory_ratio = 0.98;
    block_manager_config.host_allocator_config.blocks_num = 0;
    block_manager_config.device_allocator_config.blocks_num = 10;
    env->SetBlockManagerConfig(block_manager_config);
    KLLM_LOG_DEBUG << fmt::format("block_size {}", block_manager_config.device_allocator_config.block_size);

    env->GetRuntimeConfig(runtime_config);
    runtime_config.max_batch_size = 5;
    runtime_config.max_seq_len = 20;
    runtime_config.max_step_token_num = 40;

    BlockAllocatorGroupConfig group_1_config;
    group_1_config.devices = {0};
    group_1_config.device_block_num = block_manager_config.device_allocator_config.blocks_num;
    group_1_config.host_block_num = block_manager_config.host_allocator_config.blocks_num;
    group_1_config.block_size = block_manager_config.device_allocator_config.block_size;

    BlockAllocatorManagerConfig block_allocator_manager_config;
    block_allocator_manager_config[1] = group_1_config;

    std::shared_ptr<MemoryAllocatorInterface> memory_allocator_ = std::make_shared<MemoryAllocator>();
    BlockAllocatorManager block_allocator_manager(block_allocator_manager_config, memory_allocator_, context_);
    block_allocator_group = block_allocator_manager.GetBlockAllocatorGroup(1);

    CacheManagerConfig cache_manager_config;
    cache_manager_config.block_token_num = block_manager_config.device_allocator_config.block_token_num;
    cache_manager_config.tensor_para_size = 1;
    cache_manager_config.swap_threadpool_size = 2;
    cache_manager_config.enable_prefix_caching = true;
    cache_manager = std::make_shared<PrefixCacheManager>(cache_manager_config, block_allocator_group);
  }

  void TearDown() override {
    loguru::g_stderr_verbosity = origin_stderr_verbosity;
    if (test_name.find("ForwardNewAbsorbWithFlashMla") != std::string::npos) {
      SetAbsorbWeightsType(AbsorbWeightsType::kAbsorbDisabled);
    }
  }

 protected:
  int origin_stderr_verbosity = loguru::Verbosity_MAX;
  std::string test_name;
  ModelConfig model_config;
  RuntimeConfig runtime_config;
  std::shared_ptr<BlockAllocatorGroupInterface> block_allocator_group;
  std::shared_ptr<PrefixCacheManager> cache_manager = nullptr;

  std::shared_ptr<Context> context_{nullptr};

  template <typename weight_data_type>
  void TestMlaForward(int device_id = 0) {
    SetDevice(device_id);
    std::shared_ptr<TestWeight<weight_data_type>> base_weight =
        std::make_shared<TestWeight<weight_data_type>>(model_config, runtime_config, 0, context_);
    base_weight->AddMlaWeight(device_id);

    std::shared_ptr<ksana_llm::BaseWeight> bs1 = base_weight;
    std::shared_ptr<MultiHeadLatentAttentionTestModel<weight_data_type>> test_mla_model =
        std::make_shared<MultiHeadLatentAttentionTestModel<weight_data_type>>(model_config, runtime_config, 0, context_,
                                                                              bs1);

    // ContextDecode
    SamplingConfig sampling_config;
    ForwardRequest forward;
    forward.attn_dp_group_id = 0;
    forward.cache_manager = cache_manager;
    std::vector<int> input_ids = {233, 1681};
    forward.forwarding_tokens = &input_ids;
    forward.sampling_config = &sampling_config;
    std::vector<FlexibleCachedCopyTask> flexible_cached_copy_tasks;
    forward.flexible_cached_copy_tasks = &flexible_cached_copy_tasks;
    forward.logits_buf.resize(1);
    forward.logits_buf[0] = test_mla_model->GetLogitsPtr(schedule_id);
    forward.logits_offset = 0;
    std::vector<int> input_refit_pos;
    std::vector<std::vector<float>> input_refit_embedding;
    EmbeddingSlice embedding_slice;
    embedding_slice.pos = input_refit_pos;
    embedding_slice.embeddings = input_refit_embedding;
    forward.input_refit_embedding = &embedding_slice;
    std::vector<int> block_ids;
    block_allocator_group->GetDeviceBlockAllocator(0)->AllocateBlocks(1, block_ids);
    forward.kv_cache_ptrs.resize(1);
    block_allocator_group->GetDeviceBlockAllocator(0)->GetBlockPtrs(block_ids, forward.kv_cache_ptrs[0]);
#if defined(ENABLE_ACL) || defined(ENABLE_FLASH_ATTN_WITH_CACHE)
    // for rank_0
    forward.atb_kv_cache_base_blk_ids.clear();
    forward.atb_kv_cache_base_blk_ids.resize(1);
    LlmRuntime::BuildFlatKVCacheBlkIds(model_config.num_layer, {block_ids}, forward.atb_kv_cache_base_blk_ids,
                                       cache_manager);
#endif
    test_mla_model->CommonAttention(0, bs1, true, {forward});
    forward.forwarding_tokens->push_back(321);
    forward.kv_cached_token_num = 2;
    test_mla_model->CommonAttention(0, bs1, false, {forward});
  }
};

// 运行时检测是否是 Hopper 架构的辅助函数
bool IsHopperSupported() {
#ifdef ENABLE_CUDA
  int device = -1;
  if (cudaGetDevice(&device) != cudaSuccess) {
    return false;
  }

  int major = 0, minor = 0;
  if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess) {
    return false;
  }

  int compute_capability = major * 10 + minor;
  // Hopper 架构是 SM 9.0 (compute capability 90)
  return compute_capability >= 90;
#else
  return false;
#endif
}

TEST_F(MlaTest, ForwardNewAbsorbWithFlashMlaTest) {
#ifdef ENABLE_TOPS
  GTEST_SKIP_("ZiXiao not support this test temporary.");
#endif
  // TODO(zakwang): 支持更多类型
  // 运行时检测 GPU 架构，只有在 Hopper 架构时才运行测试
  if (!IsHopperSupported()) {
    GTEST_SKIP_("Test requires FA3 support or Hopper architecture (SM >= 9.0)");
    return;
  }
  // fp16 forward
  model_config.is_quant = false;
  model_config.weight_data_type = TYPE_FP16;
  runtime_config.inter_data_type = model_config.weight_data_type;
  model_config.quant_config.method = QUANT_NONE;
  std::cout << "Test TYPE_FP16 weight_data_type forward." << std::endl;
  TestMlaForward<float16>();
  return;
}

TEST_F(MlaTest, ForwardTest) {
#ifdef ENABLE_TOPS
  GTEST_SKIP_("ZiXiao not support this test temporary.");
#endif

  // 运行时检测 GPU 架构，只有在 Hopper 架构时才运行测试
  if (!IsHopperSupported()) {
    GTEST_SKIP_("Test requires FA3 support or Hopper architecture (SM >= 9.0)");
    return;
  }

  // fp16 forward
  model_config.is_quant = false;
  model_config.weight_data_type = TYPE_FP16;
  runtime_config.inter_data_type = model_config.weight_data_type;
  model_config.quant_config.method = QUANT_NONE;
  std::cout << "Test TYPE_FP16 weight_data_type forward." << std::endl;
  TestMlaForward<float16>();
  return;
}

TEST_F(MlaTest, ForwardNewAbsorbWithFlashMlaKvFP8Test) {
#ifdef ENABLE_TOPS
  GTEST_SKIP_("ZiXiao not support this test temporary.");
#endif

  // 运行时检测 GPU 架构，只有在支持 FA3 或 Hopper 架构时才运行测试
  if (!IsHopperSupported()) {
    GTEST_SKIP_("Test requires FA3 support or Hopper architecture (SM >= 9.0)");
    return;
  }
  // fp16 forward
  model_config.is_quant = false;
  model_config.weight_data_type = TYPE_FP16;
  runtime_config.inter_data_type = model_config.weight_data_type;
  model_config.quant_config.method = QUANT_NONE;
  runtime_config.attn_backend_config.kv_cache_dtype = TYPE_FP8_E4M3;
  std::cout << "Test TYPE_FP16 weight_data_type forward." << std::endl;
  TestMlaForward<float16>();
  return;
}
