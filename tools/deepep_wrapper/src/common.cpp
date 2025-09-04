/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "common.h"

namespace deepep_wrapper {

IPCData* CreateSharedMemory(int node_rank) {
  // 创建共享内存
  std::string shm_name = std::string(SHM_NAME) + "_" + std::to_string(node_rank);
  int shm_fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR, 0666);
  if (shm_fd == -1) {
    std::cerr << "Failed to create shared memory" << std::endl;
    return nullptr;
  }

  // 设置共享内存大小
  if (ftruncate(shm_fd, sizeof(IPCData)) == -1) {
    std::cerr << "Failed to set shared memory size" << std::endl;
    close(shm_fd);
    shm_unlink(shm_name.c_str());
    return nullptr;
  }

  // 映射共享内存
  void* addr = mmap(nullptr, sizeof(IPCData), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
  close(shm_fd);

  if (addr == MAP_FAILED) {
    std::cerr << "Failed to map shared memory" << std::endl;
    shm_unlink(shm_name.c_str());
    return nullptr;
  }

  IPCData* shared_data = static_cast<IPCData*>(addr);

  // 初始化共享数据
  memset(shared_data, 0, sizeof(IPCData));
  for (int i = 0; i < kMaxNumRanks; ++i) {
    shared_data->ipc_handle_ready[i] = false;
    shared_data->unique_id_ready[i] = false;
  }
  shared_data->ready = false;
  shared_data->error_code = 0;
  shared_data->error_message[0] = '\0';

  std::cout << "Shared memory created and initialized" << std::endl;
  return shared_data;
}

IPCData* OpenSharedMemory(int node_rank) {
  std::string shm_name = std::string(SHM_NAME) + "_" + std::to_string(node_rank);

  // Step 1: 尝试打开共享内存
  int shm_fd = shm_open(shm_name.c_str(), O_RDWR, 0666);
  if (shm_fd == -1) {
    // 共享内存还不存在
    return nullptr;
  }

  // Step 2: 检查共享内存大小是否正确
  struct stat shm_stat;
  if (fstat(shm_fd, &shm_stat) == -1) {
    close(shm_fd);
    std::cerr << "fstat(shm_fd) failed" << std::endl;
    return nullptr;
  }

  if (shm_stat.st_size < sizeof(IPCData)) {
    // 大小不足，说明还在初始化中
    close(shm_fd);
    std::cerr << "fstat(shm_fd).size() < IPCData Failed" << std::endl;
    return nullptr;
  }

  // Step 3: 映射共享内存
  void* addr = mmap(nullptr, sizeof(IPCData), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
  close(shm_fd);  // 映射后就可以关闭文件描述符

  if (addr == MAP_FAILED) {
    std::cerr << "addr == MAP_FAILED" << std::endl;
    return nullptr;
  }

  // Step 4: 安全地检查共享内存是否完全初始化
  IPCData* data = static_cast<IPCData*>(addr);

  // 使用 volatile 确保每次都从内存读取
  volatile IPCData* volatile_data = data;

  bool is_ready = false;
  bool access_safe = true;

  try {
    // 添加内存屏障，确保读取的一致性
    __sync_synchronize();

    // 检查 ready 标志
    is_ready = volatile_data->ready;
  } catch (...) {
    // 访问异常，说明还没初始化完成
    access_safe = false;
    std::cerr << "access safe Failed" << std::endl;
  }

  if (!access_safe || !is_ready) {
    // 还没准备好，释放映射
    munmap(addr, sizeof(IPCData));
    std::cerr << "Not ready" << std::endl;
    return nullptr;
  }

  // 成功！返回可用的共享内存
  std::cout << "Load Success" << std::endl;
  return data;
}

void CleanupSharedMemory(int node_rank) {
  std::string shm_name = std::string(SHM_NAME) + "_" + std::to_string(node_rank);
  shm_unlink(shm_name.c_str());
}

void GetDeviceInfo(const std::string& info) {
  CUDA_CHECK(cudaDeviceSynchronize());
  int device_id;
  CUDA_CHECK(cudaGetDevice(&device_id));

  cudaMemPool_t mempool;
  CUDA_CHECK(cudaDeviceGetDefaultMemPool(&mempool, device_id));
  size_t mempool_used;
  size_t mempool_reserved;
  CUDA_CHECK(cudaMemPoolGetAttribute(mempool, cudaMemPoolAttrUsedMemCurrent, &mempool_used));
  CUDA_CHECK(cudaMemPoolGetAttribute(mempool, cudaMemPoolAttrReservedMemCurrent, &mempool_reserved));
  size_t mempool_free = (mempool_reserved - mempool_used);
  size_t device_free;
  size_t free, total;
  CUDA_CHECK(cudaMemGetInfo(&device_free, &total));
  free = device_free + mempool_free;
  if (device_id == 0) {
    std::cout << "DeepEP " << info << " Free = " << free << std::endl;
  }
}

bool InitCuda(int rank) {
  // 检查CUDA设备数量
  int device_count;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess) {
    std::cerr << "Failed to get CUDA device count: " << cudaGetErrorString(err) << std::endl;
    return false;
  }

  if (device_count == 0) {
    std::cerr << "No CUDA devices found" << std::endl;
    return false;
  }

  // 设置CUDA设备（多GPU情况下根据rank选择）
  int device_id = rank % device_count;
  err = cudaSetDevice(device_id);
  if (err != cudaSuccess) {
    std::cerr << "Failed to set CUDA device " << device_id << ": " << cudaGetErrorString(err) << std::endl;
    return false;
  }

  // 初始化CUDA上下文
  err = cudaFree(0);
  if (err != cudaSuccess) {
    std::cerr << "Failed to initialize CUDA context: " << cudaGetErrorString(err) << std::endl;
    return false;
  }

  // 获取设备属性
  cudaDeviceProp prop;
  err = cudaGetDeviceProperties(&prop, device_id);
  if (err == cudaSuccess) {
    std::cout << "Using CUDA device " << device_id << ": " << prop.name << std::endl;
  }

  return true;
}

}  // namespace deepep_wrapper