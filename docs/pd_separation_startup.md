# PD分离启动指南

本文档介绍如何使用提供的配置文件启动KsanaLLM的PD分离（Prefill-Decode Disaggregation）服务。

## 目录

- [1. 前置准备](#1-前置准备)
- [2. 配置文件](#2-配置文件)
- [3. 启动步骤](#3-启动步骤)
- [4. 常见问题](#4-常见问题)

---

## 1. 前置准备

### 1.1 环境要求

- **操作系统**：Linux
- **GPU**：NVIDIA GPU（支持CUDA）
- **Python**：3.10+
- **依赖库**：已安装KsanaLLM及其依赖

### 1.2 必需的环境变量

启动PD分离服务时，**必须**设置以下环境变量：

```bash
export SELECT_ALL_REDUCE_BY_SIZE=1
export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync
export ENABLE_LAYER_TRACKER=1
```

**环境变量说明**：

| 环境变量 | 作用 | 必需性 |
|---------|------|--------|
| `SELECT_ALL_REDUCE_BY_SIZE` | 根据数据大小选择AllReduce算法，优化通信性能 | ✅ 必需 |
| `PYTORCH_CUDA_ALLOC_CONF` | 配置PyTorch使用cudaMallocAsync内存分配器，提升内存管理效率 | ✅ 必需 |
| `ENABLE_LAYER_TRACKER` | 启用层级追踪功能，用于PD分离的层级传输管理 | ✅ 必需 |

---

## 2. 配置文件

本指南使用以下两个配置文件：

- **Prefill配置**：`examples/ksana_llm_disaggregating_prefill.yaml`
- **Decode配置**：`examples/ksana_llm_disaggregating_decode.yaml`

### 2.1 配置文件修改

在启动前，需要修改配置文件中的以下占位符：

1. **{ROUTER_IP}**：Router服务的IP地址
2. **{PREFILL_IP}**：Prefill节点的IP地址  
3. **{DECODE_IP}**：Decode节点的IP地址
4. **model_dir**：模型文件的实际路径

**示例**：

```yaml
# 单机部署示例
connector:
  router_addr: "127.0.0.1:9080"
  inference_addr: "127.0.0.1:8067"  # Prefill
  # 或
  inference_addr: "127.0.0.1:6890"  # Decode

# 多机部署示例
connector:
  router_addr: "192.168.1.100:9080"
  inference_addr: "192.168.1.101:8067"  # Prefill
  # 或
  inference_addr: "192.168.1.102:6890"  # Decode
```

---

## 3. 启动步骤

### 3.1 启动Router服务

首先启动Router服务，用于协调Prefill和Decode节点：

```bash
# 设置环境变量
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# 启动Router服务（默认端口9080）
python src/ksana_llm/python/simple_router/main.py --port 9080
```

### 3.2 启动Prefill节点

在新的终端中启动Prefill节点：

```bash
# 设置必需的环境变量
export SELECT_ALL_REDUCE_BY_SIZE=1
export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync
export ENABLE_LAYER_TRACKER=1

# 启动Prefill服务
python src/ksana_llm/python/serving_server.py \
  --config_file examples/ksana_llm_disaggregating_prefill.yaml \
  --port 8067 \
  --host 0.0.0.0
```

**注意事项**：
- 确保配置文件中的端口（8067）与启动命令中的端口一致

### 3.3 启动Decode节点

在另一个新的终端中启动Decode节点：

```bash
# 设置必需的环境变量
export SELECT_ALL_REDUCE_BY_SIZE=1
export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync
export ENABLE_LAYER_TRACKER=1

# 启动Decode服务
python src/ksana_llm/python/serving_server.py \
  --config_file examples/ksana_llm_disaggregating_decode.yaml \
  --port 6890 \
  --host 0.0.0.0
```

**注意事项**：
- 确保配置文件中的端口（6890）与启动命令中的端口一致

### 3.4 启动顺序总结

正确的启动顺序为：

1. **Router** → 2. **Prefill节点** 和 **Decode节点**

---


## 4. 常见问题

### 4.1 网络连接问题

**症状**：节点无法连接到Router，或节点间通信失败

**解决方法**：
1. **首先检查并关闭HTTP/HTTPS代理**：
   ```bash
   # 临时关闭代理
   unset http_proxy
   unset https_proxy
   unset HTTP_PROXY
   unset HTTPS_PROXY
   
   # 或者在启动脚本中添加
   export http_proxy=""
   export https_proxy=""
   ```
2. 检查网络连接和防火墙设置
3. 确认各节点之间的网络可达性：`ping <目标IP>`

**注意**：HTTP代理可能会干扰节点间的直接通信，导致注册失败或心跳超时。如果遇到网络相关问题，建议先关闭代理再尝试。

### 4.2 节点无法注册到Router

**症状**：启动Prefill或Decode节点后，无法在Router中看到节点信息

**解决方法**：
1. 参考4.1检查并关闭HTTP代理
2. 检查Router服务是否正常运行
3. 检查配置文件中的`router_addr`是否正确
4. 检查网络连接和防火墙设置
5. 查看节点日志中的错误信息

### 4.3 环境变量未设置

**症状**：启动失败或运行异常

**解决方法**：
确保在启动Prefill和Decode节点前，已经设置了所有必需的环境变量：

```bash
export SELECT_ALL_REDUCE_BY_SIZE=1
export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync
export ENABLE_LAYER_TRACKER=1
```

### 4.4 端口冲突

**症状**：启动时提示端口已被占用

**解决方法**：
1. 检查端口是否被其他进程占用：`netstat -tulpn | grep <端口号>`
2. 修改配置文件中的端口号，确保与启动命令一致
3. 确保Prefill（8067）、Decode（6890）、Router（9080）使用不同端口

### 4.5 NCCL通信失败

**症状**：日志中出现NCCL相关错误

**解决方法**：
1. 单机部署时设置：`export NCCL_P2P_DISABLE=1`
2. 检查GPU是否可见：`nvidia-smi`
3. 确保NCCL版本与CUDA版本兼容
4. 查看NCCL详细日志：`export NCCL_DEBUG=INFO`

### 4.6 模型路径错误

**症状**：启动时提示找不到模型文件

**解决方法**：
1. 检查配置文件中的`model_dir`路径是否正确
2. 确保模型文件存在且有读取权限
3. 使用绝对路径指定模型目录
