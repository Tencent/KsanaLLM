/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/utils/device_types.h"
#include "test.h"

#include "3rdparty/half/include/half.hpp"
#include "ksana_llm/kernels/trans_layout.h"
#include "ksana_llm/utils/context.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/tensor.h"

namespace ksana_llm {

class TransLayoutTest : public testing::Test {
 protected:
  void SetUp() override {
    context = std::make_shared<Context>(1, 1, 1);
    std::filesystem::path current_path = __FILE__;
    std::filesystem::path parent_path = current_path.parent_path();
    std::filesystem::path config_path_relate = parent_path / "../../../../examples/llama7b/ksana_llm.yaml";
    std::string config_path = std::filesystem::absolute(config_path_relate).string();

    Singleton<Environment>::GetInstance()->ParseConfig(config_path);
    Singleton<Environment>::GetInstance()->GetModelConfig("", model_config);

    BlockManagerConfig block_manager_config;
    Singleton<Environment>::GetInstance()->InitializeBlockManagerConfig();
    Singleton<Environment>::GetInstance()->GetBlockManagerConfig(block_manager_config);
    KLLM_LOG_DEBUG << fmt::format("block_size {}", block_manager_config.device_allocator_config.block_size);
  }

  void TearDown() override {}

 protected:
  ModelConfig model_config;

  std::shared_ptr<Context> context{nullptr};
  int default_rank{0};
};

TEST_F(TransLayoutTest, CommonTest) {
  constexpr size_t m0 = 16;
  constexpr size_t n0 = 16;
  constexpr size_t default_batch_size = 1;
  constexpr size_t nz_default_dims = 4;
  // Expect: input content unchanged, shape from [batch, m, n] to [batch, n1, m1m0, n0] or [m, n] to [1, n1, m1m0, n0]
  // and format type change from ND to NZ.
  std::vector<std::vector<size_t>> shapes = {{32, 64}, {32, 32, 64}};
  for (const auto& shape : shapes) {
    Tensor input_dev_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, TYPE_FP16, shape, default_rank);
    std::vector<half_float::half> input_host(input_dev_tensor.GetElementNumber(), static_cast<half_float::half>(0.f));
    std::vector<half_float::half> output_host(input_dev_tensor.GetElementNumber(), static_cast<half_float::half>(0.f));

    for (size_t idx = 0; idx < input_dev_tensor.GetElementNumber(); ++idx) {
      input_host[idx] = static_cast<half_float::half>(std::sin(static_cast<float>(idx)));
    }

    MemcpyAsync(input_dev_tensor.GetPtr<void>(), input_host.data(), input_dev_tensor.GetTotalBytes(),
                MEMCPY_HOST_TO_DEVICE, context->GetMemoryManageStreams()[default_rank]);
    StreamSynchronize(context->GetMemoryManageStreams()[default_rank]);
    TransLayout(input_dev_tensor, context->GetMemoryManageStreams()[default_rank]);
    MemcpyAsync(output_host.data(), input_dev_tensor.GetPtr<void>(), input_dev_tensor.GetTotalBytes(),
                MEMCPY_DEVICE_TO_HOST, context->GetMemoryManageStreams()[default_rank]);
    StreamSynchronize(context->GetMemoryManageStreams()[default_rank]);

    size_t m = 0;
    size_t n = 0;

    EXPECT_EQ(input_dev_tensor.shape.size(), nz_default_dims);
    if (shape.size() == 2) {
      EXPECT_EQ(input_dev_tensor.shape[0], default_batch_size);
      m = shape[0];
      n = shape[1];
    }
    if (shape.size() == 3) {
      EXPECT_EQ(input_dev_tensor.shape[0], shape[0]);
      m = shape[1];
      n = shape[2];
    }
    EXPECT_EQ(input_dev_tensor.shape[1], n / n0);
    EXPECT_EQ(input_dev_tensor.shape[2], (m + m0 - 1) / m0 * m0);
    EXPECT_EQ(input_dev_tensor.shape[3], n0);
    EXPECT_EQ(input_dev_tensor.data_format, FORMAT_NZ);
    for (size_t ori_x_idx = 0; ori_x_idx < input_dev_tensor.shape[2] / m0; ++ori_x_idx) {
      for (size_t ori_y_idx = 0; ori_y_idx < input_dev_tensor.shape[1]; ++ori_y_idx) {
        for (size_t inner_x_idx = 0; inner_x_idx < m0; ++inner_x_idx) {
          for (size_t inner_y_idx = 0; inner_y_idx < n0; ++inner_y_idx) {
            EXPECT_EQ(
                static_cast<float>(output_host[ori_x_idx * n * m0 + ori_y_idx * n0 + inner_x_idx * n + inner_y_idx]),
                static_cast<float>(output_host[ori_x_idx * n * m0 + ori_y_idx * n0 + inner_x_idx * n + inner_y_idx]));
          }
        }
      }
    }
  }
}

}  // namespace ksana_llm
