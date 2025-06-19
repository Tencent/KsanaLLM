#!/bin/bash
DIR=$1
OUTPUT_DIR=$2

Candidate_BLOCK_SIZE_M=(16 32 64)
for BM in "${Candidate_BLOCK_SIZE_M[@]}"; do
    python $DIR/fused_moe_gptq_int4_fp8_kernel.py \
        --BLOCK_SIZE_M $BM --BLOCK_SIZE_N 128 --BLOCK_SIZE_K 128 --GROUP_SIZE_M 1 \
        --MUL_ROUTED_WEIGHT "False" --top_k 8 \
        --compute_type "FP16" \
        --group_size 128 \
        --m 222 --n 7168 --k 2048 --num_experts 256 \
        --output_dir $OUTPUT_DIR --tune

    python $DIR/fused_moe_gptq_int4_fp8_kernel.py \
        --BLOCK_SIZE_M $BM --BLOCK_SIZE_N 128 --BLOCK_SIZE_K 128 --GROUP_SIZE_M 1 \
        --MUL_ROUTED_WEIGHT "True" --top_k 1 \
        --compute_type "FP16" \
        --group_size 128 \
        --m 222 --n 7168 --k 2048 --num_experts 256 \
        --output_dir $OUTPUT_DIR --tune
    
    python $DIR/fused_moe_gptq_int4_fp8_kernel.py \
        --BLOCK_SIZE_M $BM --BLOCK_SIZE_N 128 --BLOCK_SIZE_K 128 --GROUP_SIZE_M 1 \
        --MUL_ROUTED_WEIGHT "False" --top_k 8 \
        --compute_type "BF16" \
        --group_size 128 \
        --m 222 --n 7168 --k 2048 --num_experts 256 \
        --output_dir $OUTPUT_DIR --tune

    python $DIR/fused_moe_gptq_int4_fp8_kernel.py \
        --BLOCK_SIZE_M $BM --BLOCK_SIZE_N 128 --BLOCK_SIZE_K 128 --GROUP_SIZE_M 1 \
        --MUL_ROUTED_WEIGHT "True" --top_k 1 \
        --compute_type "BF16" \
        --group_size 128 \
        --m 222 --n 7168 --k 2048 --num_experts 256 \
        --output_dir $OUTPUT_DIR --tune
done