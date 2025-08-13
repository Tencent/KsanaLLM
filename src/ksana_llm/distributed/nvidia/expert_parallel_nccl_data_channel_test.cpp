/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <gtest/gtest.h>

#include <arpa/inet.h>
#include <unistd.h>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ksana_llm/cache_manager/prefix_cache_manager_test_helper.h"
#include "ksana_llm/data_hub/data_hub.h"
#include "ksana_llm/data_hub/expert_data_hub.h"
#include "ksana_llm/data_hub/expert_parallel_data_transfer.h"
#include "ksana_llm/distributed/control_channel.h"

#include "ksana_llm/distributed/data_channel.h"
#include "ksana_llm/distributed/nvidia/expert_parallel_nccl_data_channel.h"
#include "ksana_llm/distributed/nvidia/nccl_data_channel.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/memory_utils.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/socket_util.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/string_utils.h"
#include "test.h"

using namespace ksana_llm;

// TODO(robertyuan): nccl only transmitted tensor, no multi_batch_id. Use default value.
#define TEST_MULTI_BATCH_ID DEFAULT_MULTI_BATCH_ID

class ExpertParallelNcclDataChannelTest : public testing::Test {
 protected:
  void SetUp() override {}

  void Initialize() {
    // Set model config.
    std::filesystem::path current_path = __FILE__;
    std::filesystem::path parent_path = current_path.parent_path();
    std::filesystem::path config_path_relate = parent_path / "../../../../examples/llama7b/ksana_llm.yaml";
    std::string config_path = std::filesystem::absolute(config_path_relate).string();

    Singleton<Environment>::GetInstance()->ParseConfig(config_path);
    env_ = Singleton<Environment>::GetInstance();
    env_->GetModelConfig(model_config_);

    // Set block manager.
    BlockManagerConfig block_manager_config;
    Singleton<Environment>::GetInstance()->InitializeBlockManagerConfig();
    Singleton<Environment>::GetInstance()->GetBlockManagerConfig(block_manager_config);

    // Init expert parallel.
    env_->GetExpertParallelConfig(expert_parallel_config_);
    expert_parallel_config_.expert_world_size = 2;
    expert_parallel_config_.expert_para_size = 1;
    expert_parallel_config_.global_expert_para_size = 2;
    expert_parallel_config_.nccl_unique_ids.resize(2);
    env_->SetExpertParallelConfig(expert_parallel_config_);

    env_->GetBlockManagerConfig(block_manager_config);
    block_manager_config.reserved_device_memory_ratio = 0.8;
    env_->SetBlockManagerConfig(block_manager_config);

    env_->GetRuntimeConfig(runtime_config_);
  }

  void TearDown() override {}

 protected:
  std::shared_ptr<Environment> env_ = nullptr;
  HiddenUnitBufferPool* hidden_unit_buffer_pool_ = nullptr;
  ExpertParallelHiddenUnitBufferPool* expert_hidden_unit_buffer_pool_ = nullptr;
  std::shared_ptr<Context> context_ = nullptr;

  std::shared_ptr<ExpertParallelNcclDataChannel> nccl_data_channel_ = nullptr;
  PipelineConfig pipeline_config_;
  ExpertParallelConfig expert_parallel_config_;
  ModelConfig model_config_;
  RuntimeConfig runtime_config_;
  std::shared_ptr<ExpertParallelDataTransfer> ep_data_transfer_;
  // The model input information.
  std::shared_ptr<ModelInput> model_input_;
};

TEST_F(ExpertParallelNcclDataChannelTest, TestDataChannel) {
  int fd[2];
  int ret = pipe(fd);
  if (ret == -1) {
    throw std::runtime_error("Create pipe error.");
  }

  char unique_id_1[128];
  char unique_id_2[128];

  const char* all_devices = getenv("CUDA_VISIBLE_DEVICES");
  if (all_devices == nullptr) {
    all_devices = "0,1";
  }
  std::vector<std::string> devices = Str2Vector(all_devices, ",");

  pid_t pid = fork();

  // Master node of expert parallel cluster.
  if (pid > 0) {
    close(fd[0]);

    setenv("CUDA_VISIBLE_DEVICES", devices[0].c_str(), 1);

    Initialize();

    // Get nccl unique_id from expert_parallel_config.
    env_->GetExpertParallelConfig(expert_parallel_config_);
    expert_parallel_config_.expert_node_rank = 0;
    expert_parallel_config_.nccl_unique_ids.resize(2);
    env_->SetExpertParallelConfig(expert_parallel_config_);

    int tp_para = runtime_config_.parallel_basic_config.tensor_parallel_size;
    uint32_t attn_data_parallel_size = runtime_config_.parallel_basic_config.attn_data_parallel_size;
    size_t multi_batch_num = 1;
    context_ = std::make_shared<Context>(tp_para, attn_data_parallel_size, multi_batch_num);

    // model_config, rank, context.
    model_input_ = std::make_shared<ModelInput>(model_config_, runtime_config_, 0, context_);
    model_input_->infer_stage == InferStage::STAGE_CONTEXT;

    // Must initialized before create data channel instance.
    InitializeHiddenUnitBufferPool();
    hidden_unit_buffer_pool_ = GetHiddenUnitBufferPool();
    InitializeExpertHiddenUnitBufferPool();
    expert_hidden_unit_buffer_pool_ = GetExpertHiddenUnitBufferPool();
    GetExpertHiddenUnitBufferPool()->SetCommType(DistributedCommunicationType::SCATTER);
    ep_data_transfer_ = std::make_shared<ExpertParallelDataTransfer>();

    nccl_data_channel_ =
        std::make_shared<ExpertParallelNcclDataChannel>(expert_hidden_unit_buffer_pool_, env_, context_);

    // Create send_thread_
    ForwardingContext forwarding_context = ForwardingContext();
    forwarding_context.SetCurrentRank(0);
    forwarding_context.SetContext(context_);
    forwarding_context.GetModelInput() = model_input_;

    // Create unique id and set to pipeline config.
    nccl_data_channel_->Listen();

    // Send nccl unique_id to child process.
    env_->GetExpertParallelConfig(expert_parallel_config_);
    ret = write(fd[1], reinterpret_cast<char*>(expert_parallel_config_.nccl_unique_ids[0].data()), 128);
    if (ret < 0) {
      throw std::runtime_error("Write pipe error.");
    }

    ret = write(fd[1], expert_parallel_config_.nccl_unique_ids[1].data(), 128);
    if (ret < 0) {
      throw std::runtime_error("Write pipe error.");
    }
    env_->SetExpertParallelConfig(expert_parallel_config_);

    close(fd[1]);

    nccl_data_channel_->Connect();

    // Get a device buffer
    HiddenUnitDeviceBuffer* dev_hidden_unit = expert_hidden_unit_buffer_pool_->GetDeviceBufferSingle();

    Tensor& tensor = dev_hidden_unit->tensors[0];
    std::vector<float> buffer_data(8, 3.14);
    Memcpy(tensor.GetPtr<void>(), buffer_data.data(), buffer_data.size() * sizeof(float), MEMCPY_HOST_TO_DEVICE);

    std::vector<Tensor> src_buffer;
    src_buffer.push_back(tensor);

    ep_data_transfer_->SendHiddenUnitBufferForEP(src_buffer, forwarding_context, true);

    CUDA_CHECK(cudaDeviceSynchronize());

    // Notify send thread to exit
    nccl_data_channel_->SetTerminate(true);
    nccl_data_channel_.reset();

  } else {
    close(fd[1]);

    setenv("CUDA_VISIBLE_DEVICES", devices[1].c_str(), 1);

    Initialize();

    // Write nccl unique_id to pipeline_config.
    env_->GetExpertParallelConfig(expert_parallel_config_);
    expert_parallel_config_.expert_world_size = 2;
    expert_parallel_config_.expert_node_rank = 1;
    expert_parallel_config_.expert_para_size = 1;
    expert_parallel_config_.global_expert_para_size = 2;
    expert_parallel_config_.nccl_unique_ids.resize(2);
    env_->SetExpertParallelConfig(expert_parallel_config_);

    int tp_para = runtime_config_.parallel_basic_config.tensor_parallel_size;
    uint32_t attn_data_parallel_size = runtime_config_.parallel_basic_config.attn_data_parallel_size;
    size_t multi_batch_num = 1;
    context_ = std::make_shared<Context>(tp_para, attn_data_parallel_size, multi_batch_num);

    // model_config, rank, context.
    model_input_ = std::make_shared<ModelInput>(model_config_, runtime_config_, 0, context_);
    model_input_->infer_stage == InferStage::STAGE_CONTEXT;

    // Must initialized before create data channel instance.
    InitializeHiddenUnitBufferPool();
    hidden_unit_buffer_pool_ = GetHiddenUnitBufferPool();
    InitializeExpertHiddenUnitBufferPool();
    expert_hidden_unit_buffer_pool_ = GetExpertHiddenUnitBufferPool();
    GetExpertHiddenUnitBufferPool()->SetCommType(DistributedCommunicationType::SCATTER);
    ep_data_transfer_ = std::make_shared<ExpertParallelDataTransfer>();

    // Recv nccl unique_id from parent process.
    memset(unique_id_1, 0, 128);
    memset(unique_id_2, 0, 128);
    ret = read(fd[0], unique_id_1, 128);
    if (ret < 0) {
      throw std::runtime_error("Read pipe error.");
    }
    ret = read(fd[0], unique_id_2, 128);
    if (ret < 0) {
      throw std::runtime_error("Read pipe error.");
    }
    close(fd[0]);

    // Write nccl unique_id to pipeline_config.
    env_->GetExpertParallelConfig(expert_parallel_config_);

    memcpy(expert_parallel_config_.nccl_unique_ids[0].data(), unique_id_1, 128);
    memcpy(expert_parallel_config_.nccl_unique_ids[1].data(), unique_id_2, 128);
    env_->SetExpertParallelConfig(expert_parallel_config_);

    nccl_data_channel_ =
        std::make_shared<ExpertParallelNcclDataChannel>(expert_hidden_unit_buffer_pool_, env_, context_);

    nccl_data_channel_->Connect();
    ForwardingContext forwarding_context = ForwardingContext();
    forwarding_context.SetCurrentRank(0);
    forwarding_context.SetContext(context_);
    forwarding_context.GetModelInput() = model_input_;

    // forwarding_context.expert_parallel_config = expert_parallel_config_;
    std::vector<Tensor> recv_tensor = ep_data_transfer_->RecvHiddenUnitBufferForEP(forwarding_context);

    // Recv from upstream
    // HiddenUnitDeviceBuffer* dev_hidden_unit = expert_hidden_unit_buffer_pool_->GetFromDeviceRecvQueue();

    CUDA_CHECK(cudaDeviceSynchronize());

    Tensor& tensor = recv_tensor[0];
    std::vector<float> buffer_data(8, 0.0);
    Memcpy(buffer_data.data(), tensor.GetPtr<void>(), buffer_data.size() * sizeof(float), MEMCPY_DEVICE_TO_HOST);

    for (auto v : buffer_data) {
      EXPECT_FLOAT_EQ(v, 3.14);
    }
    nccl_data_channel_->SetTerminate(true);
    expert_hidden_unit_buffer_pool_->Stop();
    nccl_data_channel_.reset();

#ifdef ENABLE_CUDA
    ncclDataType_t nccl_dtype;
    DataType data_type = DataType::TYPE_INT8;
    nccl_data_channel_->GetNcclDataType(data_type, nccl_dtype);
    data_type = DataType::TYPE_INT32;
    nccl_data_channel_->GetNcclDataType(data_type, nccl_dtype);
    data_type = DataType::TYPE_UINT32;
    nccl_data_channel_->GetNcclDataType(data_type, nccl_dtype);
    data_type = DataType::TYPE_UINT64;
    nccl_data_channel_->GetNcclDataType(data_type, nccl_dtype);
    data_type = DataType::TYPE_BF16;
    nccl_data_channel_->GetNcclDataType(data_type, nccl_dtype);
    data_type = DataType::TYPE_FP16;
    nccl_data_channel_->GetNcclDataType(data_type, nccl_dtype);
    data_type = DataType::TYPE_FP32;
    nccl_data_channel_->GetNcclDataType(data_type, nccl_dtype);
#endif
  }
}
