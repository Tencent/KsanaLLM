# 量化支持情况
> - S：支持
> - N：不支持
> - TODO：正在开发

## w4a16

| **量化方法** | **GPTQ/AWQ** | **GPTQ/AWQ** | **GPTQ** |
| :---------------------: | :-------------: | :-------------: | :-------------: |
| **gemm算子** | **cutlass** | **marlin** | **machete** |
| **moe算子** | **/** | **/** | **triton** |
| **硬件要求** | **sm80+** | **sm80+** | **sm90** |
| **激活形式** | **fp16/bf16** | **fp16/bf16** | **fp16/bf16** |
| **权重形式** | **group-wise int4** | **group-wise int4** | **group-wise int4** |
| **llama** | S | S | N |
| **qwen-dense** | S | S | N |
| **qwen-moe** | S | N | N |
| **deepseek/kimi** | N | N | S |

## w4a8

| **量化方法** | **W4A8_AWQ** | **RTN (moe int4)** | **RTN (moe int4)** |
| :---------------------: | :-------------: | :-----------------------------: | :-------------: |
| **gemm算子** | **cutlass** | **deepgemm(w8a8)** | **deepgemm(w8a8)** |
| **moe算子** | **/** | **triton** | **cutlass** |
| **硬件要求** | **sm89+** | **sm90** | **sm90** |
| **激活形式** | **per-tensor fp8** | **per-tensor fp8 / block-wise fp8** | **per-tensor fp8** |
| **权重形式** | **group-wise int4** | **group-wise int4** | **group-wise int4** |
| **llama** | N | N | N |
| **qwen-dense** | S | N | N |
| **qwen-moe** | N | TODO | TODO |
| **deepseek/kimi** | N | S | S |

## w8a8

| **量化方法** | **block-wise fp8** | **per-tensor fp8** |
| :---------------------: | :------------: | :------------: |
| **gemm算子** | **deepgemm** | **cublas** |
| **moe算子** | **triton** | **/** |
| **硬件要求** | **sm90** | **sm89+** |
| **激活形式** | **block-wise fp8** | **per-tensor fp8** |
| **权重形式** | **block-wise fp8** | **per-tensor fp8** |
| **llama** | N | S |
| **qwen-dense** | N | S |
| **qwen-moe** | N | S |
| **deepseek/kimi** | S | N |


## 说明

### 量化方法

- **GPTQ** 基于近似二阶信息的后训练量化（PTQ）方法，利用 Hessian 矩阵补偿量化误差，实现高精度低比特量化。
- **AWQ** 感知激活分布的权重量化方法，通过保留少量关键权重（基于激活幅度）的精度，显著减少性能损失。
- **RTN** 直接将浮点数值舍入到最近的量化点，简单高效，但在低比特下精度损失较大。
- **W4A8_AWQ** 结合 4-bit 权重（W4）和 8-bit 激活（A8）的混合精度方案，通常利用 AWQ 优化权重，平衡计算效率和显存占用。

### 量化粒度

- **Per-Tensor** 整个张量共享一组量化参数，计算开销最小，但精度损失可能较大。
- **Group-wise** 将张量划分为多个组（如 128 个连续元素），通常为一维（1D）划分，每组独立量化，平衡了精度和开销。
- **Block-wise** 类似于 Group-wise，但通常指二维（2D）划分（如 128x128 块），每块共享缩放因子，能有效应对异常值，提升精度。
