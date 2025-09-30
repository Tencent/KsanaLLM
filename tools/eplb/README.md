# EPLB (Expert-Parallel Load Balancer) 背景

- 使用 Expert-Parallel 专家并行时，不同的专家会被分配在不同的 GPU 卡上。以 DeepSeek-R1 的 256 个专家， EP8 来举例:
  - 默认情况下，我们顺序在 8 张卡上，均匀地加载 256 个专家，每张卡上加载 32 个专家（左闭右开表示）：
    - 卡0 加载专家 [0, 32)
    - 卡1 加载专家 [32, 64)
    - 卡2 加载专家 [64, 96)
    - ...
    - 卡7 加载专家 [224, 256)
  - 而由于输入的不同，因此事实上不同专家被激活的次数和频率均有不同，也就是我们常说的有 “热门专家” 的存在。
- 由于专家被激活的不均匀，在 MOE 层依据 topk_ids 执行 AllToAll 通信后，有可能会出现部分节点上要计算的 token 数较多，二有部分节点上要计算的 token 数较少的问题。最极端的情况，**某些节点可能根本没有要计算的 token**。节点间待计算的 token 数量不同，会导致一部分节点空闲，由于 MOE 后还需要同步的调用 Combine 做第二次数据交换，这会使一部分节点出现较长时间等待，降低整体系统的吞吐。

# 静态 EPLB

## 映射表数据结构

- 使用二维数组来存储各层的 EPLB 专家映射表：expert_map[layer_num][num_experts]
  - 第一维为总层数，由于各层专家激活分布不同，因此每层都应有一个对应的映射表
  - 第二维为总专家数，用于将 [0, num_experts) 范围内的专家序号，映射为实际显存中存放的专家序号

## 专家分布文件生成

- 一念服务启动时，通过 环境变量 控制，将 TopkIds Tensor 使用 SaveToNpyFile dump 到本地目录
  - ```
    ENABLE_DUMP_EPLB_DATA=1 \
    DUMP_EPLB_PATH=/tmp/dump/ \
    python ../src/ksana_llm/python/serving_server.py \
      --config_file ksana_llm.yaml \
      --port 6789
    ```
  - 其中: DUMP_EPLB_PATH 可不传,默认将存储在 ${HOME}/.cache/KsanaLLM/EPLB/ 目录下
- 启动一念后,正常完成待分析样本的请求和推理过程
- 在配置的目录下，将自动生成对应的 dump 数据, dump 数据会按照层为单位，存储在不同的目录下.

## 映射表生成

- 使用脚本 KsanaLLM/tools/eplb/expert_parallel_get_map.py，将上一节生成的专家分布文件，转换为 json 映射表
  - ```
    python tools/eplb/expert_parallel_get_map.py \
		-i /tmp/dump/ \
        -o eplb_config.json \
        --ep_nums 256 \
        --nodes 1 \
        --gpu_per_nodes 8 \
        -s 1 \
        -e 26 \
        --max_files 1024 \
        --perf_optimize
    ```
  - 下面详细说明脚本的各项入参含义：
    - --input_dir，-i，指向的是上一节中的环境变量所在地址 KSANA_DUMP_TOPK_PATH
    - --output_file, -o，表示的是输出的 json 文件名称
    - --ep_nums，总的专家数
    - --nodes，总的节点数（多机为N，单机为1）
    - --gpu_per_nodes，每台机器上的卡数（若使用单机4卡，则这里应该写4）
    - --start_layer，-s，可不传，存在 MOE 的最小层 id，不传则根据给定目录自动分析
    - --end_layer，-e，可不传，存在 MOE 的最大层 id，不传则根据给定目录自动分析
    - --max_files，可不传，用于生成映射表的最大文件数，避免测试集过大生成较慢
    - --perf_optimize，-p，开启后会在执行路径生成一个仿真模拟结果，预估静态 EPLB 在当前数据集的性能提升效果。

## 使用映射表

- 一念服务启动时，通过 环境变量 控制，将静态 EPLB 专家映射表 load 到一念程序中
- ```
    export EPLB_WEIGHT=/work/KsanaLLM/build/eplb_config.json
    python ../src/ksana_llm/python/serving_server.py \
        --config_file ksana_llm.yaml \
        --port 6789
  ```
