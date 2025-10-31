/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/modules/basic/bmm.h"
#include "ksana_llm/models/base/fake_weight_for_test.h"
#include "ksana_llm/models/common/common_weight.h"
#include "ksana_llm/utils/context.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/memory_allocator.h"
#include "ksana_llm/utils/tensor.h"
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
#ifdef ENABLE_BFLOAT16
GET_KSANA_DATA_TYPE(__nv_bfloat16, TYPE_BF16);
#endif
#undef GET_KSANA_DATA_TYPE

void AssignFromVector(Tensor& tensor, const std::vector<float>& f_vector) {
  DeviceSynchronize();
  int device_rank;
  GetDevice(&device_rank);

  if (f_vector.size() != tensor.GetElementNumber()) {
    KLLM_THROW("Vector size does not match tensor element count");
  }

  torch::Tensor cpu_tensor =
      torch::from_blob(const_cast<float*>(f_vector.data()), {static_cast<int64_t>(f_vector.size())}, torch::kFloat32);

  DataType dtype_impl = tensor.dtype;
  auto options = torch::TensorOptions().device(torch::kCUDA, device_rank).dtype(GetTorchTypeFromDataType(dtype_impl));

  void* tensor_data_ptr = tensor.GetPtr<void>();
  torch::Tensor gpu_tensor =
      torch::from_blob(tensor_data_ptr, {static_cast<int64_t>(tensor.GetElementNumber())}, options);
  gpu_tensor.copy_(cpu_tensor.to(options.device()));

  DeviceSynchronize();
}

void RandomizeInputData(std::vector<float>& input_data1, size_t tokens, size_t heads, size_t qk_nope_dim) {
  size_t total_elements = tokens * heads * qk_nope_dim;
  CustomRandomGenerator generator;
  generator.FillNormal(input_data1, total_elements, 1.0, 0.5);
}

// Batched matrix multiplication: output = input_data * weight
template <typename T>
void BatchedMatmul(const std::vector<T>& input_data, const std::vector<T>& weight, std::vector<T>& output,
                   size_t tokens, size_t heads, size_t qk_nope_dim, size_t kv_lora_rank) {
  assert(output.size() == heads * tokens * kv_lora_rank);

  for (size_t h = 0; h < heads; ++h) {             // heads
    for (size_t t = 0; t < tokens; ++t) {          // tokens
      for (size_t n = 0; n < kv_lora_rank; ++n) {  // kv_lora_rank
        T sum = 0;
        for (size_t k = 0; k < qk_nope_dim; ++k) {  // qk_nope_dim
          // input_data[tokens, heads, qk_nope_dim] 乘 weight[heads, qk_nope_dim, kv_lora_rank]
          sum += input_data[t * heads * qk_nope_dim + h * qk_nope_dim + k] *
                 weight[h * qk_nope_dim * kv_lora_rank + k * kv_lora_rank + n];
        }
        // output[tokens, heads, kv_lora_rank]
        output[t * heads * kv_lora_rank + h * kv_lora_rank + n] = sum;
      }
    }
  }
}

float GetAllowedDiff(const std::string& test_type) {
  if (test_type == "find_algo") {
    return 5e-07;
  } else if (test_type == "half_type") {
    return 0.0007;
  } else if (test_type == "float_type") {
    return 2e-07;
  } else if (test_type == "head_e1") {
    return 0.0006;
  } else if (test_type == "tokens_e1") {
    return 0.0005;
  } else if (test_type == "heads_e_tokens") {
    return 2e-07;
  }
  return 0.01f;
}

float CalcDiff(float gpu_val, float cpu_val) {
  if (cpu_val == 0.0f) {
    return std::fabs(gpu_val);
  }
  return std::fabs((gpu_val - cpu_val) / cpu_val);
}

template <typename T>
void TestBmmWithType(size_t heads, size_t qk_nope_dim, size_t kv_lora_rank, size_t tokens, std::string test_type) {
  // Initialize the relevant
  ModelConfig model_config;
  model_config.weight_data_type = GetKsanaDataType<T>();

  RuntimeConfig runtime_config;
  runtime_config.inter_data_type = GetKsanaDataType<T>();

  int rank = 0;
  auto ctx = std::make_shared<Context>(1, 1, 1);

  // weight
  size_t elem_count1 = heads * qk_nope_dim * kv_lora_rank;
  size_t byte_size1 = elem_count1 * sizeof(T);
  GTEST_LOG_(INFO) << "typesize: " << sizeof(T);

  // input
  size_t elem_count2 = heads * qk_nope_dim * tokens;
  size_t byte_size2 = elem_count2 * sizeof(T);

  // output
  size_t elem_count3 = kv_lora_rank * tokens * heads;
  size_t byte_size3 = elem_count3 * sizeof(T);

  // Create bmm_weight and Tensor
  auto bmm_weight = std::make_shared<CommonWeight<T>>(model_config, runtime_config, rank, ctx);

  void* new_buf = nullptr;
  cudaError_t e = cudaMalloc(&new_buf, byte_size1);
  ASSERT_EQ(e, cudaSuccess) << "CUDA memory allocation failed for gpu_buf: " << cudaGetErrorString(e);

  Tensor new_tensor(MemoryLocation::LOCATION_DEVICE, GetKsanaDataType<T>(), {heads, qk_nope_dim, kv_lora_rank}, 0,
                    new_buf);

  std::vector<float> input_data(new_tensor.GetElementNumber());
  RandomizeInputData(input_data, heads, qk_nope_dim, kv_lora_rank);
  AssignFromVector(new_tensor, input_data);

  // Initialize the layer weights
  bmm_weight->weights_map_[""] = new_tensor;

  auto matmul_factory = std::make_shared<MatMulLayerFactory>(model_config, runtime_config, rank, ctx);
  auto pipeline_config = std::make_shared<PipelineConfig>();
  BufferManager buffer_mgr;
  buffer_mgr.SetRank(rank);

  LinearComputeBackend backend = DEFAULT_LINEAR_BACKEND;
  LayerCreationContext creation_ctx;
  creation_ctx.Init(bmm_weight, ctx, rank, *pipeline_config, model_config, runtime_config, &buffer_mgr);

  Bmm my_bmm("", creation_ctx, backend);

  // Allocate space for the input and assign real values
  void* gpu_buf = nullptr;
  cudaError_t err = cudaMalloc(&gpu_buf, byte_size2);
  ASSERT_EQ(err, cudaSuccess) << "CUDA memory allocation failed for gpu_buf: " << cudaGetErrorString(err);
  Tensor gpu_tensor(MemoryLocation::LOCATION_DEVICE, GetKsanaDataType<T>(), {tokens, heads, qk_nope_dim}, 0, gpu_buf);
  std::vector<float> input_data1(tokens * heads * qk_nope_dim);
  RandomizeInputData(input_data1, heads, tokens, qk_nope_dim);
  AssignFromVector(gpu_tensor, input_data1);

  // The space used for calculating the output
  void* gpu_buf2 = nullptr;
  cudaError_t err2 = cudaMalloc(&gpu_buf2, byte_size3);
  ASSERT_EQ(err2, cudaSuccess) << "CUDA memory allocation failed for gpu_buf1: " << cudaGetErrorString(err2);
  Tensor gpu_tensor2(MemoryLocation::LOCATION_DEVICE, GetKsanaDataType<T>(), {tokens, heads, kv_lora_rank}, 0,
                     gpu_buf2);

  std::vector<Tensor> input_tensors = {gpu_tensor};
  std::vector<Tensor> output_tensors = {gpu_tensor2};

  Status status = my_bmm.Forward(input_tensors, output_tensors);

  std::vector<T> res(output_tensors[0].GetElementNumber());

  Memcpy(res.data(), output_tensors[0].template GetPtr<void>(), sizeof(T) * output_tensors[0].GetElementNumber(),
         MEMCPY_DEVICE_TO_HOST);

  std::vector<float> output(output_tensors[0].GetElementNumber());
  BatchedMatmul<float>(input_data1, input_data, output, tokens, heads, qk_nope_dim, kv_lora_rank);

  float diff = 0;
  float max_diff = 0;
  float thresh = GetAllowedDiff(test_type);
  for (size_t i = 0; i < output.size(); i++) {
    diff = CalcDiff(static_cast<float>(res[i]), output[i]);
    EXPECT_TRUE(diff < thresh);
    max_diff = (diff > max_diff ? diff : max_diff);
  }
  std::cerr << max_diff << std::endl;
}

// half test
TEST(BmmInitTest, half_type) {
  std::string test_type = "half_type";
  size_t heads = 4, qk_nope_dim = 4, kv_lora_rank = 2, tokens = 8;
  TestBmmWithType<half>(heads, qk_nope_dim, kv_lora_rank, tokens, test_type);
}

// float test
TEST(BmmInitTest, float_type) {
  std::string test_type = "float_type";
  size_t heads = 4, qk_nope_dim = 4, kv_lora_rank = 2, tokens = 8;
  TestBmmWithType<float>(heads, qk_nope_dim, kv_lora_rank, tokens, test_type);
}

// head=1 test
TEST(BmmInitTest, head_e1) {
  std::string test_type = "head_e1";
  size_t heads = 1, qk_nope_dim = 4, kv_lora_rank = 2, tokens = 8;
  TestBmmWithType<half>(heads, qk_nope_dim, kv_lora_rank, tokens, test_type);
}

// tokens=1 test
TEST(BmmInitTest, tokens_e1) {
  std::string test_type = "tokens_e1";
  size_t heads = 8, qk_nope_dim = 12, kv_lora_rank = 20, tokens = 1;
  TestBmmWithType<half>(heads, qk_nope_dim, kv_lora_rank, tokens, test_type);
}

// find algo
TEST(BmmInitTest, find_algo) {
  std::string test_type = "find_algo";
  size_t heads = 8, qk_nope_dim = 32, kv_lora_rank = 128, tokens = 6000;
  TestBmmWithType<float>(heads, qk_nope_dim, kv_lora_rank, tokens, test_type);
}
