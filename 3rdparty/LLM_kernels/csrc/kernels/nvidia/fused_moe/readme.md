# fused_moe.py Usage Guide

### --tune（Basic Usage）
Search for the optimal `num_warps` and `num_stages` based on the MoE parameters, then generate Triton kernel.
```bash
python ./fused_moe.py \
    --output_dir ./ \
    --group_n 128 \
    --group_k 128 \
    --BLOCK_SIZE_M 64 \
    --BLOCK_SIZE_N 128 \
    --BLOCK_SIZE_K 128 \
    --GROUP_SIZE_M 32 \
    --MUL_ROUTED_WEIGHT "True" \
    --top_k 1 \
    --compute_type "FP16" \
    --use_fp8_w8a8 "True" \
    --use_int8_w8a16 "False" \
    --tune
```

### --model_path
Reads parameters from the specified model config to override input parameters (`k`, `n`, `num_experts`, `topk`, `block_shape`) 

### --deep_tune
Optimizes configurations (`BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K`, `GROUP_SIZE_M`, `num_warps`, `num_stages`). No Triton kernel will be generated in this mode; instead, a JSON file with the best configuration will be saved as `best_config.json`.

```bash
{
    "128": {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 3
    }
}
```

In this scenario, since the fused MoE kernel operates in two steps (the first with `MUL_ROUTED_WEIGHT` set to False, and the second set to True), the `MUL_ROUTED_WEIGHT` parameter is ignored. Additionally, `deep_tune` can only be enabled when `tune` is set to True.

```bash
python ./fused_moe.py \
    --output_dir ./ \
    --group_n 128 \
    --group_k 128 \
    --BLOCK_SIZE_M 64 \
    --BLOCK_SIZE_N 128 \
    --BLOCK_SIZE_K 128 \
    --GROUP_SIZE_M 32 \
    --MUL_ROUTED_WEIGHT "False" \
    --top_k 8 \
    --compute_type "FP16" \
    --use_fp8_w8a8 "True" \
    --use_int8_w8a16 "False" \
    --tune \
    --deep_tune \
    --model_path "/model/DeepSeek" \
    --tp_size 8 \
    --m 20 \
    --even_Ks True
```

### --tp_size
Simulates multi-GPU parallelism (`n // tp_size`)