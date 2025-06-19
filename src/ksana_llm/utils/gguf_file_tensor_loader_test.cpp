/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/utils/gguf_file_tensor_loader.h"
#include <filesystem>
#include <fstream>
#include "gtest/gtest.h"
#include "ksana_llm/utils/device_types.h"

// Namespace alias for convenience
namespace ksana = ksana_llm;

class GGUFLoadTest : public ::testing::Test {
 protected:
  // Path to the GGUF test file
  std::string test_file_name_ = "test_model.gguf";

  // Pointer to the GGUFFileTensorLoader instance
  std::unique_ptr<ksana::GGUFFileTensorLoader> tensor_loader_;

  // This function will be called before each test is run.
  void SetUp() override {
    // Create a test GGUF file
    CreateTestGGUFFile();

    // Initialize the GGUFFileTensorLoader with the test file
    tensor_loader_ = std::make_unique<ksana::GGUFFileTensorLoader>(test_file_name_, true);
  }

  // This function will be called after each test is run.
  void TearDown() override {
    // Delete the test GGUF file
    if (std::filesystem::exists(test_file_name_)) {
      std::filesystem::remove(test_file_name_);
    }
  }

  // Helper function to create a minimal GGUF file for testing
  void CreateTestGGUFFile() {
    std::ofstream outfile(test_file_name_, std::ios::binary);

    // Write GGUF magic number
    uint32_t magic = GGUF_MAGIC;
    outfile.write(reinterpret_cast<char*>(&magic), sizeof(uint32_t));

    // Write GGUF version
    uint32_t version = GGUF_VERSION;
    outfile.write(reinterpret_cast<char*>(&version), sizeof(uint32_t));

    // Write tensor count (1 tensor)
    uint64_t tensor_count = 1;
    outfile.write(reinterpret_cast<char*>(&tensor_count), sizeof(uint64_t));

    // Write metadata key-value count (0 for simplicity)
    uint64_t metadata_kv_count = 0;
    outfile.write(reinterpret_cast<char*>(&metadata_kv_count), sizeof(uint64_t));

    // Begin tensor information
    // Tensor name length and name
    std::string tensor_name = "test_tensor";
    uint64_t tensor_name_length = tensor_name.size();
    outfile.write(reinterpret_cast<char*>(&tensor_name_length), sizeof(uint64_t));
    outfile.write(tensor_name.c_str(), tensor_name_length);

    // Number of dimensions (2D tensor)
    uint32_t n_dims = 2;
    outfile.write(reinterpret_cast<char*>(&n_dims), sizeof(uint32_t));

    // Dimensions
    uint64_t dim0 = 3;
    uint64_t dim1 = 4;
    outfile.write(reinterpret_cast<char*>(&dim0), sizeof(uint64_t));
    outfile.write(reinterpret_cast<char*>(&dim1), sizeof(uint64_t));

    // Data type (GGML_TYPE_F32)
    uint32_t data_type = ksana::GGMLType::GGML_TYPE_F32;
    outfile.write(reinterpret_cast<char*>(&data_type), sizeof(uint32_t));

    // Offset (must be aligned to GGUF_ALIGNMENT)
    uint64_t offset = 0;  // Placeholder, will be adjusted after header
    outfile.write(reinterpret_cast<char*>(&offset), sizeof(uint64_t));

    // Calculate the offset and adjust for alignment
    size_t header_size = outfile.tellp();
    size_t alignment = GGUF_ALIGNMENT;
    size_t padding = (alignment - (header_size % alignment)) % alignment;
    for (size_t i = 0; i < padding; ++i) {
      outfile.put(0);
    }

    // Update the offset
    offset = header_size + padding;
    outfile.seekp(header_size - sizeof(uint64_t), std::ios::beg);
    outfile.write(reinterpret_cast<char*>(&offset), sizeof(uint64_t));
    outfile.seekp(0, std::ios::end);

    // Write tensor data (initialize with zeros for simplicity)
    size_t tensor_size = dim0 * dim1 * sizeof(float);
    std::vector<float> tensor_data(dim0 * dim1, 0.0f);
    outfile.write(reinterpret_cast<char*>(tensor_data.data()), tensor_size);

    outfile.close();
  }
};

// Test case to check if tensor names are loaded correctly
TEST_F(GGUFLoadTest, GetTensorNameListTest) {
  const std::vector<std::string>& tensor_names = tensor_loader_->GetTensorNameList();
  ASSERT_EQ(tensor_names.size(), 1);
  EXPECT_EQ(tensor_names[0], "test_tensor");
}

// Test case to check if tensor data can be retrieved correctly
TEST_F(GGUFLoadTest, GetTensorDataTest) {
  void* tensor_ptr;
  size_t tensor_size;

  std::tie(tensor_ptr, tensor_size) = tensor_loader_->GetTensor("test_tensor");

  ASSERT_NE(tensor_ptr, nullptr);
  EXPECT_EQ(tensor_size, 3 * 4 * sizeof(float));
}

// Test case to check if tensor shape is correct
TEST_F(GGUFLoadTest, GetTensorShapeTest) {
  std::vector<size_t> shape = tensor_loader_->GetTensorShape("test_tensor");
  ASSERT_EQ(shape.size(), 2);
  EXPECT_EQ(shape[0], 4);
  EXPECT_EQ(shape[1], 3);
}

// Test case to check if data type is correct
TEST_F(GGUFLoadTest, GetTensorDataTypeTest) {
  ksana::DataType data_type = tensor_loader_->GetTensorDataType("test_tensor");
  EXPECT_EQ(data_type, ksana::TYPE_FP32);
}

// Test case to check if invalid tensor name returns null
TEST_F(GGUFLoadTest, GetInvalidTensorTest) {
  void* tensor_ptr;
  size_t tensor_size;

  std::tie(tensor_ptr, tensor_size) = tensor_loader_->GetTensor("invalid_tensor");

  EXPECT_EQ(tensor_ptr, nullptr);
  EXPECT_EQ(tensor_size, 0);
}

// Test case to check the metadata (should be empty)
TEST_F(GGUFLoadTest, GetMetadataTest) {
  auto context = tensor_loader_->GetMetadata();
  ASSERT_NE(context, nullptr);
  EXPECT_EQ(context->metadata_map.size(), 0);
}

// Test case to check tensor setting functionality
TEST_F(GGUFLoadTest, SetTensorTest) {
  // Create a new tensor with the same shape
  torch::Tensor tensor = torch::zeros({3, 4}, torch::dtype(torch::kFloat32));

  // Set tensor
  tensor_loader_->SetTensor("test_tensor", tensor);

  // Retrieve tensor pointer
  void* tensor_ptr;
  size_t tensor_size;

  std::tie(tensor_ptr, tensor_size) = tensor_loader_->GetTensor("test_tensor");

  ASSERT_NE(tensor_ptr, nullptr);
  EXPECT_EQ(tensor_size, 3 * 4 * sizeof(float));

  // Verify that the data pointer matches
  EXPECT_EQ(tensor_ptr, tensor.data_ptr());
}

// Test case to check file name retrieval
TEST_F(GGUFLoadTest, GetTensorFileNameTest) {
  std::string file_name = tensor_loader_->GetTensorFileName();
  EXPECT_EQ(file_name, test_file_name_);
}

// Test case to check handling of missing GGUF file
TEST_F(GGUFLoadTest, MissingFileTest) {
  // Delete the test file
  std::filesystem::remove(test_file_name_);

  // Expect the constructor to throw an exception
  EXPECT_THROW({ ksana::GGUFFileTensorLoader loader("nonexistent_file.gguf", true); }, std::runtime_error);
}

// Test case to check alignment handling
TEST_F(GGUFLoadTest, AlignmentTest) {
  auto context = tensor_loader_->GetMetadata();
  ASSERT_NE(context, nullptr);
  EXPECT_EQ(context->alignment, GGUF_ALIGNMENT);
}

// Test case to check ConvertGGMLTypeToDataType function (indirectly via data type check)
TEST_F(GGUFLoadTest, ConvertGGMLTypeToDataTypeTest) {
  ksana::DataType data_type = tensor_loader_->GetTensorDataType("test_tensor");
  EXPECT_EQ(data_type, ksana::TYPE_FP32);
}

// Test case to check handling when tensor count is zero
TEST_F(GGUFLoadTest, ZeroTensorCountTest) {
  // Create a GGUF file with zero tensors
  std::string zero_tensor_file = "zero_tensor.gguf";
  {
    std::ofstream outfile(zero_tensor_file, std::ios::binary);

    // Write GGUF magic number
    uint32_t magic = GGUF_MAGIC;
    outfile.write(reinterpret_cast<char*>(&magic), sizeof(uint32_t));

    // Write GGUF version
    uint32_t version = GGUF_VERSION;
    outfile.write(reinterpret_cast<char*>(&version), sizeof(uint32_t));

    // Write tensor count (0 tensors)
    uint64_t tensor_count = 0;
    outfile.write(reinterpret_cast<char*>(&tensor_count), sizeof(uint64_t));

    // Write metadata key-value count (0)
    uint64_t metadata_kv_count = 0;
    outfile.write(reinterpret_cast<char*>(&metadata_kv_count), sizeof(uint64_t));

    outfile.close();
  }

  // Load the zero-tensor GGUF file
  ksana::GGUFFileTensorLoader loader(zero_tensor_file, true);

  // Check that the tensor name list is empty
  const std::vector<std::string>& tensor_names = loader.GetTensorNameList();
  EXPECT_EQ(tensor_names.size(), 0);

  // Clean up
  std::filesystem::remove(zero_tensor_file);
}