# 一念LLM (KsanaLLM)

[English](README.md) [中文](README_cn.md)

## 介绍

**一念LLM** 是面向LLM推理和服务的高性能和高易用的推理引擎。

**高性能和高吞吐:**

- 使用极致优化的 CUDA kernels, 包括来自 [vLLM](https://github.com/vllm-project/vllm), [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), [FastTransformer](https://github.com/NVIDIA/FasterTransformer), [SGLang](https://github.com/sgl-project/sglang), [LightLLM](https://github.com/ModelTC/lightllm) 等工作的高性能算子
- 基于 [PagedAttention](https://arxiv.org/abs/2309.06180) 实现地对注意力机制中key和value的高效显存管理
- 对任务调度和显存占用精细调优的动态batching
- 支持前缀缓存(Prefix caching)
- 支持DeepSeek-MTP
- 在A10, A100, L20, L40, H20, 910B2C等卡上做了较充分的验证测试

**灵活易用:**

- 能够无缝集成流行的 Hugging Face 格式模型，支持 PyTorch 和 SafeTensor 两种权重格式

- 能够实现高吞吐服务，支持多种解码算法，包括并行采样、beam search 等

- 支持多卡间的 tensor 并行 

- 支持流式输出

- 支持 OpenAI-compatible API server

- 支持英伟达 GPU 和华为昇腾 NPU


**一念LLM 支持 Hugging Face 的很多流行模型，下面是经过验证测试的模型:**

- LLaMA 7B/13B & LLaMA-2 7B/13B & LLaMA3 8B/70B
- Baichuan1 7B/13B & Baichuan2 7B/13B
- Qwen 7B/14B & QWen1.5 7B/14B/72B/110B QWen-VL
- Yi1.5-34B
- DeepSeek V3/R1

**支持的硬件**

 - Nvidia GPUs: A10, A100, L40, L20, H20
 - Huawei Ascend NPUs: 910B2C

## 使用

### 1. 创建 Docker 容器和运行时环境

#### 1.1 英伟达 GPU

```bash
# 请先根据https://github.com/NVIDIA/nvidia-container-toolkit安装nvidia-docker
cd docker
docker build -f Dockerfile.gpu -t ksanallm-gpu .
docker run \
    -u root \
    -itd --privileged \
    --shm-size=50g \
    --network host \
    --cap-add=SYS_ADMIN \
    --cap-add=SYS_PTRACE \
    --gpus all \
    ksanallm-gpu bash

# 进入KsanaLLM根目录 
pip install -r requirements.txt
```

#### 1.2 直接使用腾讯云GPU镜像

```bash
# need install nvidia-docker from https://github.com/NVIDIA/nvidia-container-toolkit
cd docker
nvidia-docker build -f Dockerfile.tencentos4.gpu -t ksanallm-gpu .
nvidia-docker run \
    -u root \
    -itd --privileged \
    --shm-size=50g \
    --network host \
    --cap-add=SYS_ADMIN \
    --cap-add=SYS_PTRACE \
    ksanallm-gpu bash

# goto KsanaLLM root directory 
pip install -r requirements.txt
```

#### 1.3 华为昇腾 NPU

**请先安装Huawei Ascend NPU驱动和CANN：[驱动下载链接](https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/softwareinst/instg/instg_0000.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit)**

**推荐版本：CANN 8.0RC2**

**目前只支持华为昇腾NPU + X86 CPU**

```bash
cd docker
docker build -f Dockerfile.npu -t ksanallm-npu .
docker run \
    -u root \
    -itd --privileged \
    --shm-size=50g \
    --network host \
    --cap-add=SYS_ADMIN \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp:unconfined $(find /dev/ -regex ".*/davinci$" | awk '{print " --device "$0}') \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
    -v /usr/local/sbin/:/usr/local/sbin/ \
    -v /var/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf \
    -v /var/log/npu/slog/:/var/log/npu/slog \
    -v /var/log/npu/profiling/:/var/log/npu/profiling \
    -v /var/log/npu/dump/:/var/log/npu/dump \
    -v /var/log/npu/:/usr/slog \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    ksanallm-npu bash

# 从下面链接安装Ascend-cann-toolkit, Ascend-cann-nnal：https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/softwareinst/instg/instg_0000.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit
# 从下面链接下载torch_npu-2.1.0.post6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl：https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/softwareinst/instg/instg_0000.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit
pip3 install torch==2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install torch_npu-2.1.0.post6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
pip install -r requirements.txt
```

### 2. 下载源码

```bash
git clone --recurse-submodules https://github.com/pcg-mlp/KsanaLLM
export GIT_PROJECT_REPO_ROOT=`pwd`/KsanaLLM
```

### 3. 编译

```bash
cd ${GIT_PROJECT_REPO_ROOT}
pip install -r requirements.txt
mkdir build && cd build
```

#### 3.1 英伟达 GPU

```bash
# SM for A10 is 86， change it when using other gpus.
# refer to: https://developer.nvidia.cn/cuda-gpus
cmake -DSM=86 -DWITH_TESTING=ON .. && make -j32
```

#### 3.2 华为昇腾 NPU

```bash
cmake -DWITH_TESTING=ON -DWITH_CUDA=OFF -DWITH_ACL=ON .. && make -j32
```

#### 3.3 如果需要使用腾讯内部的依赖库，例如北极星名字服务，需要开启以下选项:

```bash
cmake -DSM=86 -DWITH_TESTING=ON -DWITH_INTERNAL_LIBRARIES=ON .. && make -j32
```

### 4. 执行

#### 4.1 单机执行

```bash
cd ${GIT_PROJECT_REPO_ROOT}/src/ksana_llm/python
ln -s ${GIT_PROJECT_REPO_ROOT}/build/lib .

# download huggingface model for example:
git clone https://huggingface.co/NousResearch/Llama-2-7b-hf

# change the model_dir in ${GIT_PROJECT_REPO_ROOT}/examples/ksana_llm2-7b.yaml if needed

# set environment variable `KLLM_LOG_LEVEL=DEBUG` before run to get more log info
# the serving log locate in log/ksana_llm.log

# ${GIT_PROJECT_REPO_ROOT}/examples/ksana_llm2-7b.yaml's tensor_para_size equal the GPUs/NPUs number
export CUDA_VISIBLE_DEVICES=xx

# launch server
python serving_server.py \
    --config_file ${GIT_PROJECT_REPO_ROOT}/examples/ksana_llm2-7b.yaml \
    --port 8080

#KsanaLLM 正在支持 OpenAI API 协议，目前已经支持了主要的/v1/chat/completions API
# 在启动服务时可以添加对应的参数以支持工具调用解析和推理内容解析能力
# 工具解析：
--enable-auto-tool-choice     --tool-call-parser deepseek_v3  \
# 推理内容解析:
--reasoning-parser  deepseek_r1 \
#应用特定的 chat-template
--chat-template openaiapi/chat_templates/tool_chat_template_deepseekr1.jinja
```

基于one shot对话的推理测试 

```bash
# open another session
cd ${GIT_PROJECT_REPO_ROOT}/src/ksana_llm/python
python serving_generate_client.py --port 8080
```

forward推理测试（单轮推理，无采样）

```bash
python serving_forward_client.py --port 8080
```

提示：KsanaLLM会在服务启动的目录下自动生成log文件，可通过以下方式查看模型加载、服务启动，请求告警等信息。
```bash
vim log/ksana_llm.log
```

#### 4.2 分布式执行

分布式执行依赖以下环境变量：
WORLD_SIZE:  结点个数，即推理进程个数，可以同机，也可以跨机。如果未定义或者值为1，则不是分布式模式
NODE_RANK:   当前结点的rank，从0开始，0为master结点
MASTER_HOST: 推理集群master结点的IP地址
MASTER_PORT: 推理集群master管理端口


下面以IP1和IP2，master节点部署在IP1，监听端口为port_1，演示双机执行的命令
```bash
# 在IP1上执行
export WORLD_SIZE=2
export NODE_RANK=0
export MASTER_HOST=IP1
export MASTER_PORT=port_1

python serving_server.py \
    --config_file ${GIT_PROJECT_REPO_ROOT}/examples/ksana_llm2-7b.yaml \
    --port 8080


# 在IP2上执行
export WORLD_SIZE=2
export NODE_RANK=1
export MASTER_HOST=IP1
export MASTER_PORT=port_1

python serving_server.py \
    --config_file ${GIT_PROJECT_REPO_ROOT}/examples/ksana_llm2-7b.yaml \
    --port 8080
```

备注：默认走NCCL通信，如果要强制走TCP通信，可以增加下面环境变量：
export USE_TCP_DATA_CHANNEL=1

#### 4.3 H20执行DeepSeek模型示例

##### 4.3.1 NVIDIA H20 编译

```bash
cmake -DSM=90a -DCMAKE_BUILD_TYPE=Release ..  && make -j

cd ${GIT_PROJECT_REPO_ROOT}/src/ksana_llm/python
ln -s ${GIT_PROJECT_REPO_ROOT}/build/lib .
```

##### 4.3.2 双机16卡H20执行（以[DeepSeek-R1-0528模型](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528)为例）
并行策略：节点间使用流水线并行，节点内使用张量并行，以下为性能最优配置

1. IP1节点（master节点）执行：
```bash
# 将IP1节点设置为master节点
export WORLD_SIZE=2
export NODE_RANK=0
export MASTER_HOST=master_node_ip
export MASTER_PORT=master_node_port
# 最优环境变量配置
export ENABLE_COMPRESSED_KV=2
export SELECT_ALL_REDUCE_BY_SIZE=1
export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync
export USE_TCP_DATA_CHANNEL=1
export MASTER_OFFLOAD_LAYER_NUM=0
# 服务启动
python serving_server.py \
--config_file ${GIT_PROJECT_REPO_ROOT}/examples/deepseek_fp8_perf.yaml \
--port service_port
```

2. IP2节点（work节点）执行：
```bash
# 将IP2节点设置为work节点
export WORLD_SIZE=2
export NODE_RANK=1
export MASTER_HOST=master_node_ip
export MASTER_PORT=master_node_port
# 最优环境变量配置
export ENABLE_COMPRESSED_KV=2
export SELECT_ALL_REDUCE_BY_SIZE=1
export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync
export USE_TCP_DATA_CHANNEL=1
# 服务启动
python serving_server.py \
--config_file ${GIT_PROJECT_REPO_ROOT}/examples/deepseek_fp8_perf.yaml \
--port service_port
```
备注：当前版本若使用mutil-batch功能(deepseek_fp8_perf.yaml中的max_pp_batch_num=2)需在节点间使用TCP(export USE_TCP_DATA_CHANNEL=1)进行通信，后续会支持NCCL通信。

3. 提升服务启动速度（可选）:

如果你觉得服务启动过慢，可以通过配置以下环境变量生成cache模型，这样在下一次服务启动的时候会加载cache模型，加快服务启动。
```bash
export ENABLE_MODEL_CACHE=1
export MODEL_CACHE_PATH=/xxx_cache_model_dir/
```
备注：生成cache模型和使用cache模型都需配置以上环境变量，且每个节点都需要配置。

##### 4.3.3 单机8卡H20执行DeepSeek-R1-0528-GPTQ-int4模型

```bash
# 最优环境变量配置
export ENABLE_COMPRESSED_KV=2
export SELECT_ALL_REDUCE_BY_SIZE=1
export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync
#（可选）进一步提升性能，略有精度损失
export W4AFP8_MOE_BACKEND=1
# 服务启动
python serving_server.py \
--config_file ${GIT_PROJECT_REPO_ROOT}/examples/deepseek_int4_perf.yaml \
--port service_port
```
备注：int4模型的服务启动也可以通过配置4.3.2中的生成cache模型的环境变量来加速

##### 4.3.4 性能压测（通用）
```bash
python ${GIT_PROJECT_REPO_ROOT}/benchmarks/benchmark_throughput.py \
--host master_node_ip \
--port service_port \
--prompt_num 512 \
--input_csv xxx_dataset.csv \
--stream \
--backend ksana \
--model_type deepseek_r1/deepseek_v3 \
--mode async \
--request_rate xx_qps \
--output_csv output_res.csv \
--perf_csv perf_res.csv
# request_rate可以控制请求发送的速率，默认为"inf"(同时发送所有请求)
```

### 5. 分发

```bash
cd ${GIT_PROJECT_REPO_ROOT}

# for distribute wheel
python setup.py bdist_wheel

# or build with other cmake args
export CMAKE_ARGS="
  -DWITH_CUDA=ON
  -DWITH_ACL=OFF
" && python setup.py bdist_wheel

# install wheel
pip install dist/ksana_llm-0.1-*-linux_x86_64.whl

# check install success
pip show -f ksana_llm
python -c "import ksana_llm"
```

### 6. 可选

#### 6.1 模型权重映射

在支持新模型时，如果模型结构与已知模型（例如Llama）相同，只是权重名字不同，可以通过JSON文件来对权重做一个映射，从而能够较简单的支持新模型。想要获取更详细的信息，请参考: [Optional Weigth Map Guide](src/ksana_llm/python/weight_map/README.md)。

#### 6.2 自定义插件

自定义插件可以做特殊预处理和后处理。使用时，你需要把`ksana_plugin.py`放在模型目录下。

你需要实现类`KsanaPlugin`，包含3个可选的方法：
`init_plugin(self, **kwargs)`, `preprocess(self, **kwargs)` and `postprocess(self, **kwargs)`。

- `init_plugin`会在插件初始化时被调用一次
- `preprocess`会在每条请求开始时被调用一次（如推理ViT）
- `postprocess`会在每条请求结束前被调用一次（如计算困惑度PPL）

更多细节见[示例](src/ksana_llm/python/ksana_plugin/qwen_vl/ksana_plugin.py)。

#### 6.3 KV Cache缩放因子

打开FP8 E4M3 KV Cache量化时，为保证推理精度需要提供缩放因子。想要获取更详细的信息，请参考: [Optional KV Scale Guide](src/ksana_llm/python/kv_scale_files/README.md)。

#### 7. 联系我们
##### 微信群
<img src=docs/img/webchat-github.jpg width="200px">
