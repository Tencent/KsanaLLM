# KsanaLLM

[English](README.md) | [中文](README_cn.md)

## About

KsanaLLM is a high performance and easy-to-use engine for LLM inference and serving.

**High Performance and Throughput:**

- Utilizes optimized CUDA kernels, including high performance kernels from [vLLM](https://github.com/vllm-project/vllm), [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), [FastTransformer](https://github.com/NVIDIA/FasterTransformer), [SGLang](https://github.com/sgl-project/sglang), [LightLLM](https://github.com/ModelTC/lightllm)
- Efficient management of attention key and value memory with [PagedAttention](https://arxiv.org/abs/2309.06180)
- Detailed optimization of task-scheduling and memory-uitlization for dynamic batching 
- Prefix caching support
- Sufficient testing has been conducted on GPU/NPU cards such as A10, A100, L20, L40, H20, 910B2C etc

**Flexibility and easy to use:**

- Seamless integration with popular Hugging Face models, and support multiple weight formats, such as pytorch and SafeTensors

- High-throughput serving with various decoding algorithms, including parallel sampling, beam search, and more

- Enables multi-gpu tensor parallelism 

- Streaming outputs

- OpenAI-compatible API server

- Support NVIDIA GPUs and Huawei Ascend NPU


**KsanaLLM seamlessly supports many Hugging Face models, including the below models that have been verified:**

- LLaMA 7B/13B & LLaMA-2 7B/13B & LLaMA3 8B/70B
- Baichuan1 7B/13B & Baichuan2 7B/13B
- Qwen 7B/14B & QWen1.5 7B/14B/72B/110B Qwen-VL
- Yi1.5-34B 
- DeepSeek V3/R1

**Supported Hardware**

 - Nvidia GPUs: A10, A100, L40, L20, H20
 - Huawei Ascend NPUs: 910B2C

## Usage

### 1. Create Docker container and runtime environment

#### 1.1 For Nvidia GPU

```bash
# need install nvidia-docker from https://github.com/NVIDIA/nvidia-container-toolkit
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

# goto KsanaLLM root directory 
pip install -r requirements.txt
```

#### 1.2 Direct Use of Tencent Cloud Nvidia GPU Image

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

#### 1.3 For Huawei Ascend NPU

**Please install Huawei Ascend NPU driver and CANN: [driver download link](https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/softwareinst/instg/instg_0000.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit)**

**Recommend version: CANN 8.0RC2**

**Only Support Ascend NPU + X86 CPU**

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

# install Ascend-cann-toolkit, Ascend-cann-nnal from https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/softwareinst/instg/instg_0000.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit
# download torch_npu-2.1.0.post6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl from https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/softwareinst/instg/instg_0000.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit
pip3 install torch==2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install torch_npu-2.1.0.post6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
pip install -r requirements.txt
```

### 2. Clone source code

```bash
git clone --recurse-submodules https://github.com/pcg-mlp/KsanaLLM
export GIT_PROJECT_REPO_ROOT=`pwd`/KsanaLLM
```

### 3. Compile

```bash
cd ${GIT_PROJECT_REPO_ROOT}
pip install -r requirements.txt
mkdir build && cd build
```

#### 3.1 For Nvidia

```bash
# SM for A10 is 86， change it when using other gpus.
# refer to: https://developer.nvidia.cn/cuda-gpus
cmake -DSM=86 -DWITH_TESTING=ON .. && make -j32
```

#### 3.2 For Huawei Ascend NPU

```bash
cmake -DWITH_TESTING=ON -DWITH_CUDA=OFF -DWITH_ACL=ON .. && make -j32
```

#### 3.3 Enable use of Tencent internal libraries, for example, Tencent nameserver Polaris

```bash
cmake -DSM=86 -DWITH_TESTING=ON -DWITH_INTERNAL_LIBRARIES=ON .. && make -j32
```

### 4. Run

#### 4.1 Single

```bash
cd ${GIT_PROJECT_REPO_ROOT}/src/ksana_llm/python
ln -s ${GIT_PROJECT_REPO_ROOT}/build/lib .

# download huggingface model for example:
# Note: Make sure git-lfs is installed.
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
# KsanaLLM now supports the OpenAI API protocol and has implemented the core /v1/chat/completions API. 
# you can add specific parameters to enable tool invocation parsing and inference content analysis capabilities.
# tool-choice:
--enable-auto-tool-choice     --tool-call-parser deepseek_v3  \
# reasoning-parser:
--reasoning-parser  deepseek_r1 \
# apply specific chat-template:
--chat-template openaiapi/chat_templates/tool_chat_template_deepseekr1.jinja

```

Tip: KsanaLLM automatically generates log files in the directory where the service is started. You can view information such as model loading, service startup, and request warning through these log files.
```bash
vim log/ksana_llm.log
```

Inference test with one shot conversation

```bash
# open another session
cd ${GIT_PROJECT_REPO_ROOT}/src/ksana_llm/python
python serving_generate_client.py --port 8080
```

Inference test with forward(Single round inference without generate sampling)

```bash
python serving_forward_client.py --port 8080
```

Test performance of the model
```bash
cd ${GIT_PROJECT_REPO_ROOT}/build 
./bin/run_model_performance --runtime-config ${GIT_PROJECT_REPO_ROOT}/examples/llama7b/ksana_llm_tp.yaml --csv-config ${GIT_PROJECT_REPO_ROOT}/src/ksana_llm/model_performance/test_config.csv 

# enable nsys when using Cuda
export ENABLE_PROFILE_EVENT=1 # enale profile event like NVTX on Cuda
nsys profile ./bin/run_model_performance --runtime-config ${GIT_PROJECT_REPO_ROOT}/examples/llama7b/ksana_llm_tp.yaml --csv-config ${GIT_PROJECT_REPO_ROOT}/src/ksana_llm/model_performance/test_config.csv 
unset ENABLE_PROFILE_EVENT # after using nsys
```

#### 4.2 Distributed

Distributed execution depends on the following environment variables:
WORLD_SIZE: Number of nodes, i.e., number of inference processes, which can be on the same machine or across machines. If undefined or the value is 1, it is not distributed mode.
NODE_RANK: The rank of the current node, starting from 0, with 0 being the master node.
MASTER_HOST: The IP address of the master node in the inference cluster.
MASTER_PORT: The management port of the master node in the inference cluster.

Below, using IP1 and IP2, with the master node deployed on IP1 and listening on port_1, demonstrates the command for dual-machine execution.
```bash
# on IP1
export WORLD_SIZE=2
export NODE_RANK=0
export MASTER_HOST=IP1
export MASTER_PORT=port_1

python serving_server.py \
    --config_file ${GIT_PROJECT_REPO_ROOT}/examples/ksana_llm2-7b.yaml \
    --port 8080


# on IP2
export WORLD_SIZE=2
export NODE_RANK=1
export MASTER_HOST=IP1
export MASTER_PORT=port_1

python serving_server.py \
    --config_file ${GIT_PROJECT_REPO_ROOT}/examples/ksana_llm2-7b.yaml \
    --port 8080
```

Note: By default, NCCL communication is used. If you want to force TCP communication, you can add the following environment variable:
export USE_TCP_DATA_CHANNEL=1

#### 4.3 Example of Running the DeepSeek Model on H20

##### 4.3.1 Compilation for NVIDIA H20

```bash
cmake -DSM=90a -DWITH_VLLM_FLASH_ATTN=ON -DCMAKE_BUILD_TYPE=Release ..  && make -j

cd ${GIT_PROJECT_REPO_ROOT}/src/ksana_llm/python
ln -s ${GIT_PROJECT_REPO_ROOT}/build/lib .
```

##### 4.3.2 Dual-Node 16-GPU Execution Example (Using the [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) Model)
Parallelization Strategy: Inter-node Pipeline Parallelism and Intra-node Tensor Parallelism—Optimal Performance Configuration as Follows

1. Execution on IP1 Node (Master Node):
```bash
# Set the IP1 node as the master node.
export WORLD_SIZE=2
export NODE_RANK=0
export MASTER_HOST=master_node_ip
export MASTER_PORT=master_node_port
# Optimal environment variable configuration
export ENABLE_COMPRESSED_KV=2
export SELECT_ALL_REDUCE_BY_SIZE=1
export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync
export USE_TCP_DATA_CHANNEL=1
export MASTER_OFFLOAD_LAYER_NUM=0
# Service Startup
python serving_server.py \
--config_file ${GIT_PROJECT_REPO_ROOT}/examples/deepseek_fp8_perf.yaml \
--port service_port
```

2. Execution on IP2 Node (Work Node):
```bash
# Set the IP2 node as the work node.
export WORLD_SIZE=2
export NODE_RANK=1
export MASTER_HOST=master_node_ip
export MASTER_PORT=master_node_port
# Optimal environment variable configuration
export ENABLE_COMPRESSED_KV=2
export SELECT_ALL_REDUCE_BY_SIZE=1
export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync
export USE_TCP_DATA_CHANNEL=1
# Service Startup
python serving_server.py \
--config_file ${GIT_PROJECT_REPO_ROOT}/examples/deepseek_fp8_perf.yaml \
--port service_port
```
Note: In the current version, when using the multi-batch feature (i.e., setting max_pp_batch_num=2 in deepseek_fp8_perf.yaml), 
inter-node communication must be conducted via TCP (by exporting USE_TCP_DATA_CHANNEL=1). NCCL-based communication for multi-batch will be supported in future releases.

3. Improving Service Startup Speed (Optional):

If you find that the service startup is too slow, you can accelerate the process by configuring the following environment variables to generate a cached model. 
Upon subsequent service startups, the cached model will be loaded, thereby reducing startup latency.
```bash
export ENABLE_MODEL_CACHE=1
export MODEL_CACHE_PATH=/xxx_cache_model_dir/
```
Note: Both the generation and utilization of the cached model require the above environment variables to be set. Additionally, these configurations must be applied on every node.

##### 4.3.3 Single-Node 8-GPU Execution of the DeepSeek-R1-0528-GPTQ-int4 Model

```bash
# Optimal environment variable configuration
export ENABLE_COMPRESSED_KV=2
export SELECT_ALL_REDUCE_BY_SIZE=1
export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync
# (Optional) Further Performance Enhancement with Slight Accuracy Degradation
export EXPERIMENTAL_INT4_FP8_MOE=1
# Service Startup
python serving_server.py \
--config_file ${GIT_PROJECT_REPO_ROOT}/examples/deepseek_int4_perf.yaml \
--port service_port
```
Note: The startup of int4 models can also be accelerated by configuring the environment variables for generating cached models as described in Section 4.3.2.

##### 4.3.4 Performance Stress Testing (General)
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
# The request_rate parameter controls the rate at which requests are sent. By default, it is set to "inf" (all requests are sent simultaneously).
```

### 5. Distribute

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

### 6. Optional 

#### 6.1 Model Weight Map

You can include an optional weight map JSON file for models that share the same structure as the Llama model but have different weight names.

For more detailed information, please refer to the following link: [Optional Weight Map Guide](src/ksana_llm/python/weight_map/README.md)

#### 6.2 Plugin

Custom plugins can perform some special pre-processing and post-processing tasks. You need to place your `ksana_plugin.py` in the
model directory.

You should implement a `KsanaPlugin` class with three optional methods:
`init_plugin(self, **kwargs)`, `preprocess(self, **kwargs)` and `postprocess(self, **kwargs)`.

- `init_plugin` is called during plugin initialization
- `preprocess` is called at the start of each request (e.g., ViT inference)
- `postprocess` is called at the end of each request (e.g., PPL calculation)

See [Example](src/ksana_llm/python/ksana_plugin/qwen_vl/ksana_plugin.py) for more details.

#### 6.3 KV Cache Scaling Factors

When enabling FP8 E4M3 KV Cache quantization, it is necessary to provide scaling factors to ensure inference accuracy.

For more detailed information, please refer to the following link: [Optional KV Scale Guide](src/ksana_llm/python/kv_scale_files/README.md)

#### 7. Contact Us
##### WeChat Group
<img src=docs/img/webchat-github.jpg width="200px">

