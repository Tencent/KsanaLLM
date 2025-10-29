# Ksana LLM 监控指标说明文档

## 概述

本文档详细说明了 Ksana LLM 项目中通过 `REPORT_METRIC` 和 `REPORT_COUNTER` 上报的所有监控指标。这些指标用于监控推理服务的性能、吞吐量、延迟等关键指标。

### 监控指标类型

- **REPORT_METRIC**: 用于上报直方图类型的指标（Histogram），记录数值的分布情况，适合延迟、耗时等指标
- **REPORT_COUNTER**: 用于上报计数器类型的指标（Counter），记录累计值，适合请求数、token数等指标

## 监控指标分类体系

### 1. 请求指标（Request Level Metrics）

#### 1.1 请求总体统计


| 指标名称                  | 类型    | 单位 | 说明                 | 计算方式               |
| --------------------------- | --------- | ------ | ---------------------- | ------------------------ |
| `forward_req_total_num`   | Counter | 个   | 接收到的推理请求总数 | 每接收一个请求累加1    |
| `forward_req_timeout_num` | Counter | 个   | 超时的请求数量       | 请求超时时累加1        |
| `forward_req_aborted_num` | Counter | 个   | 被中止的请求数量     | 请求被用户中止时累加1  |
| `forward_req_error_num`   | Metric  | -    | 错误请求的错误码分布 | 记录请求失败时的错误码 |

#### 1.2 请求延迟指标


| 指标名称                          | 类型   | 单位 | 说明                        | 计算方式                                |
| ----------------------------------- | -------- | ------ | ----------------------------- | ----------------------------------------- |
| `total_latency_us`                | Metric | 微秒 | 请求从接收到完成的总延迟    | 请求完成时间 - 请求接收时间             |
| `time_to_first_token_us`          | Metric | 微秒 | 首token延迟（TTFT）         | 第一个token生成时间 - 请求接收时间      |
| `inter_token_interval_latency_us` | Metric | 微秒 | 相邻两次token生成的时间间隔 | 当前token生成时间 - 上一个token生成时间 |
| `batch_manager_schedule_us`       | Metric | 微秒 | 请求在调度器中的等待时间    | 请求开始调度时间 - 请求接收时间         |

**延迟指标关系**：

```
total_latency_us ≈ time_to_first_token_us + (output_token_len - 1) × inter_token_interval_latency_us（平均值）
```

#### 1.3 Token 统计


| 指标名称                    | 类型    | 单位 | 说明                | 计算方式                     |
| ----------------------------- | --------- | ------ | --------------------- | ------------------------------ |
| `input_token_len`           | Metric  | 个   | 输入token长度       | 请求的输入token数量          |
| `output_token_len`          | Metric  | 个   | 输出token长度       | 请求生成的输出token数量      |

### 2. 批处理指标（Batch Level Metrics）

#### 2.1 吞吐量指标


| 指标名称                  | 类型    | 单位     | 说明                       | 计算方式                                          |
| --------------------------- | --------- | ---------- | ---------------------------- | --------------------------------------------------- |
| `global_token_throughput` | Metric  | tokens/s | 全局token吞吐量            | forwarding_token_num  / 两次forward之间的时间间隔 |
| `local_token_throughput`  | Metric  | tokens/s | 单次forward的token吞吐量   | forwarding_token_num  / forward耗时               |
| `forwarding_token_num`    | Counter | 个       | 单次forward处理的token数量 | 批次中所有请求的token数之和                       |
| `computed_token_num`        | Counter | 个   | 已计算的token总数   | 累计所有实际计算的token数    |
| `computed_input_token_num`  | Counter | 个   | 已计算的输入token数 | 累计prefill阶段计算的token数 |
| `computed_output_token_num` | Counter | 个   | 已计算的输出token数 | 累计decode阶段计算的token数  |

**吞吐量指标关系**：

```
global_token_throughput 考虑了调度等待时间，通常小于 local_token_throughput
local_token_throughput 仅考虑实际计算时间，反映了纯计算性能
对比computed_token_num的区别为这里多次调度会将速度取均值，computed_token_num是系统实际的计算个数累加更接近客户端的吞吐情况。
computed_token_num = computed_input_token_num + computed_output_token_num
```

#### 2.2 批处理时间分解



| 指标名称               | 类型    | 单位 | 说明                      | 计算方式                                                            |
| ------------------------ | --------- | ------ | --------------------------- | --------------------------------------------------------------------- |
| `schedule_time_us`     | Counter | 微秒 | 调度阶段耗时              | 调度开始到forward开始的时间                                         |
| `forwarding_time_us`   | Counter | 微秒 | Forward总耗时             | forward开始到结束的总时间                                           |
| `forwarding_time_rate` | Metric  | %    | Forward时间占总时间的比例 | forwarding_time_us / (schedule_time_us + forwarding_time_us) × 100 |

**时间分解关系**：

```
总处理时间 = schedule_time_us + forwarding_time_us
```

### 3. 调度器指标（Scheduler Level Metrics）

#### 3.1 队列状态


| 指标名称                                | 类型    | 单位 | 说明                   | 计算方式                    |
| ----------------------------------------- | --------- | ------ | ------------------------ | ----------------------------- |
| `batch_scheduler_batch_size`            | Metric  | 个   | 当前批次中的请求数量   | 正在运行的请求数            |
| `batch_scheduler_batch_size_{dp_idx}`   | Metric  | 个   | 指定DP组的批次大小     | 特定数据并行组的运行请求数  |
| `batch_scheduler_waiting_size`          | Metric  | 个   | 等待队列中的请求数量   | 等待被调度的请求数          |
| `batch_scheduler_waiting_size_{dp_idx}` | Metric  | 个   | 指定DP组的等待队列大小 | 特定数据并行组的等待请求数  |
| `batch_scheduler_swapped_size`          | Metric  | 个   | 换出队列中的请求数量   | 被换出到CPU内存的请求数     |
| `batch_scheduler_pending_swapin_size`   | Counter | 个   | 待换入的请求数量       | 等待从CPU换入GPU的请求数    |
| `batch_scheduler_pending_swapout_size`  | Counter | 个   | 待换出的请求数量       | 等待从GPU换出到CPU的请求数  |
| `num_tokens_to_schedule`                | Metric  | 个   | 待调度的token总数      | 所有待调度请求的token数之和 |

#### 3.2 调度器时间分解（微秒级）


| 指标名称                                 | 类型   | 单位 | 说明               | 计算方式                           |
| ------------------------------------------ | -------- | ------ | -------------------- | ------------------------------------ |
| `batch_scheduler_time_us`                | Metric | 微秒 | 调度器总耗时       | 调度器处理的总时间                 |
| `batch_scheduler_running_queue_time_us`  | Metric | 微秒 | 处理运行队列的耗时 | 处理正在运行的请求所需时间         |
| `batch_scheduler_swapped_queue_time_us`  | Metric | 微秒 | 处理换出队列的耗时 | 处理换出请求所需时间               |
| `batch_scheduler_waiting_queue_time_us`  | Metric | 微秒 | 处理等待队列的耗时 | 处理等待请求所需时间               |
| `batch_scheduler_transfer_queue_time_us` | Metric | 微秒 | 处理传输队列的耗时 | 处理PD分离模式下的传输请求所需时间 |

**调度器时间分解关系**：

```
batch_scheduler_time_us = 
    batch_scheduler_running_queue_time_us +
    batch_scheduler_swapped_queue_time_us +
    batch_scheduler_waiting_queue_time_us +
    batch_scheduler_transfer_queue_time_us
```

### 4. 缓存相关指标（Cache Metrics）

#### 4.1 前缀缓存（Prefix Cache）


| 指标名称                        | 类型    | 单位 | 说明                  | 计算方式                               |
| --------------------------------- | --------- | ------ | ----------------------- | ---------------------------------------- |
| `prefix_cache_hit_req_num`      | Counter | 个   | 前缀缓存命中的请求数  | 使用了前缀缓存的请求数累加             |
| `prefix_cache_hit_token_num`    | Counter | 个   | 前缀缓存命中的token数 | 通过前缀缓存复用的token数累加          |
| `prefix_cache_hit_block_num`    | Counter | 个   | 前缀缓存命中的block数 | 通过前缀缓存复用的KV cache block数累加 |
| `full_prompt_matched_req_num`   | Counter | 个   | 完全匹配的请求数      | 输入完全匹配缓存的请求数               |
| `full_prompt_matched_block_num` | Counter | 个   | 完全匹配的block数     | 完全匹配时复用的block数                |

**前缀缓存效率计算**：

```
前缀缓存命中率 = prefix_cache_hit_req_num / forward_req_total_num
平均每次命中节省的token数 = prefix_cache_hit_token_num / prefix_cache_hit_req_num
平均每次命中节省的block数 = prefix_cache_hit_block_num / prefix_cache_hit_req_num
```

### 5. 推测解码指标（Speculative Decoding Metrics）


| 指标名称               | 类型   | 单位 | 说明                | 计算方式                        |
| ------------------------ | -------- | ------ | --------------------- | --------------------------------- |
| `spec_draft_hit_num`   | Metric | 个   | 推测命中的token数   | 推测生成的token被验证通过的数量 |
| `spec_draft_token_num` | Metric | 个   | 推测生成的token总数 | draft模型生成的所有token数量    |

**推测解码效率计算**：

```
推测解码命中率 = spec_draft_hit_num / spec_draft_token_num
```

### 6. Prefill-Decode 分离模式指标（PD Separation Metrics）

#### 6.1 任务传输指标


| 指标名称                    | 类型   | 单位 | 说明                     | 计算方式                        |
| ----------------------------- | -------- | ------ | -------------------------- | --------------------------------- |
| `pd_task_total_cost_us`     | Metric | 微秒 | 任务从创建到完成的总耗时 | 任务完成时间 - 任务创建时间     |
| `pd_task_prepare_cost_us`   | Metric | 微秒 | 任务准备阶段的耗时       | 任务元信息处理的时间            |
| `pd_task_send_data_cost_us` | Metric | 微秒 | 任务数据发送的平均耗时   | 数据发送总时间 / 发送的任务组数 |
| `pd_task_recv_data_cost_us` | Metric | 微秒 | 任务数据接收的平均耗时   | 数据接收总时间 / 接收的任务数   |
| `pd_task_waiting_cost_us`   | Metric | 微秒 | 任务等待确认的耗时       | decode端确认时间 - 任务创建时间 |

#### 6.2 任务传输统计


| 指标名称                       | 类型   | 单位 | 说明                 | 计算方式                            |
| -------------------------------- | -------- | ------ | ---------------------- | ------------------------------------- |
| `pd_transfer_task_batch_size`  | Metric | 个   | 单次传输的任务数量   | 从prefill传输到decode的任务批次大小 |
| `pd_transfer_skipped_task_num` | Metric | 个   | 跳过传输的任务数量   | 因各种原因未传输的任务数            |
| `transfer_tasks_per_request`   | Metric | 个   | 每个请求的传输任务数 | 完成一个请求所需的传输任务总数      |



## 监控指标层次关系图

```
系统总体性能
├── 请求
│   ├── 请求统计（总数、超时、中止、错误）
│   ├── 延迟指标（总延迟、TTFT、token间隔）
│   └── Token统计（输入、输出、已计算）
│
├── 批处理
│   ├── 吞吐量（全局、本地、token数）
│   └── 时间分解（调度、forward、各步骤）
│
├── 调度器
│   ├── 队列状态（批次、等待、换出、swap pending）
│   └── 时间分解（总时间、各队列处理时间）
│
└── 优化特性
    ├── 前缀缓存（命中请求、token、block）
    ├── 推测解码（命中数、总数）
    └── PD分离（任务传输、耗时分解）