#!/bin/bash
DIR=$1
OUTPUT_DIR=$2

python $DIR/fused_moe_kernel_gptq_awq.py \
    --BLOCK_SIZE_M 16 --BLOCK_SIZE_N 32 --BLOCK_SIZE_K 64 --GROUP_SIZE_M 1 \
    --MUL_ROUTED_WEIGHT "False" --top_k 8 \
    --compute_type "FP16" \
    --has_zp "False" \
    --weight_bits 4 \
    --group_size 128 \
    --m 32 --n 4096 --k 7168 \
    --num_experts 256 \
    --output_dir $OUTPUT_DIR --tune

python $DIR/fused_moe_kernel_gptq_awq.py \
    --BLOCK_SIZE_M 16 --BLOCK_SIZE_N 64 --BLOCK_SIZE_K 32 --GROUP_SIZE_M 1 \
    --MUL_ROUTED_WEIGHT "False" --top_k 8 \
    --compute_type "FP16" \
    --has_zp "False" \
    --weight_bits 4 \
    --group_size 128 \
    --m 222 --n 4096 --k 7168 \
    --num_experts 256 \
    --output_dir $OUTPUT_DIR --tune

python $DIR/fused_moe_kernel_gptq_awq.py \
    --BLOCK_SIZE_M 32 --BLOCK_SIZE_N 32 --BLOCK_SIZE_K 64 --GROUP_SIZE_M 1 \
    --MUL_ROUTED_WEIGHT "False" --top_k 8 \
    --compute_type "FP16" \
    --has_zp "False" \
    --weight_bits 4 \
    --group_size 128 \
    --m 32 --n 4096 --k 7168 \
    --num_experts 256 \
    --output_dir $OUTPUT_DIR --tune

python $DIR/fused_moe_kernel_gptq_awq.py \
    --BLOCK_SIZE_M 32 --BLOCK_SIZE_N 64 --BLOCK_SIZE_K 32 --GROUP_SIZE_M 1 \
    --MUL_ROUTED_WEIGHT "False" --top_k 8 \
    --compute_type "FP16" \
    --has_zp "False" \
    --weight_bits 4 \
    --group_size 128 \
    --m 222 --n 4096 --k 7168 \
    --num_experts 256 \
    --output_dir $OUTPUT_DIR --tune

python $DIR/fused_moe_kernel_gptq_awq.py \
    --BLOCK_SIZE_M 64 --BLOCK_SIZE_N 32 --BLOCK_SIZE_K 64 --GROUP_SIZE_M 1 \
    --MUL_ROUTED_WEIGHT "False" --top_k 8 \
    --compute_type "FP16" \
    --has_zp "False" \
    --weight_bits 4 \
    --group_size 128 \
    --m 32 --n 4096 --k 7168 \
    --num_experts 256 \
    --output_dir $OUTPUT_DIR --tune

python $DIR/fused_moe_kernel_gptq_awq.py \
    --BLOCK_SIZE_M 64 --BLOCK_SIZE_N 64 --BLOCK_SIZE_K 32 --GROUP_SIZE_M 1 \
    --MUL_ROUTED_WEIGHT "False" --top_k 8 \
    --compute_type "FP16" \
    --has_zp "False" \
    --weight_bits 4 \
    --group_size 128 \
    --m 222 --n 4096 --k 7168 \
    --num_experts 256 \
    --output_dir $OUTPUT_DIR --tune

python $DIR/fused_moe_kernel_gptq_awq.py \
    --BLOCK_SIZE_M 16 --BLOCK_SIZE_N 32 --BLOCK_SIZE_K 64 --GROUP_SIZE_M 1 \
    --MUL_ROUTED_WEIGHT "True" --top_k 1 \
    --compute_type "FP16" \
    --has_zp "False" \
    --weight_bits 4 \
    --group_size 128 \
    --m 32 --n 4096 --k 7168 \
    --num_experts 256 \
    --output_dir $OUTPUT_DIR --tune

python $DIR/fused_moe_kernel_gptq_awq.py \
    --BLOCK_SIZE_M 16 --BLOCK_SIZE_N 64 --BLOCK_SIZE_K 32 --GROUP_SIZE_M 1 \
    --MUL_ROUTED_WEIGHT "True" --top_k 1 \
    --compute_type "FP16" \
    --has_zp "False" \
    --weight_bits 4 \
    --group_size 128 \
    --m 222 --n 4096 --k 7168 \
    --num_experts 256 \
    --output_dir $OUTPUT_DIR --tune

python $DIR/fused_moe_kernel_gptq_awq.py \
    --BLOCK_SIZE_M 32 --BLOCK_SIZE_N 32 --BLOCK_SIZE_K 64 --GROUP_SIZE_M 1 \
    --MUL_ROUTED_WEIGHT "True" --top_k 1 \
    --compute_type "FP16" \
    --has_zp "False" \
    --weight_bits 4 \
    --group_size 128 \
    --m 32 --n 4096 --k 7168 \
    --num_experts 256 \
    --output_dir $OUTPUT_DIR --tune

python $DIR/fused_moe_kernel_gptq_awq.py \
    --BLOCK_SIZE_M 32 --BLOCK_SIZE_N 64 --BLOCK_SIZE_K 32 --GROUP_SIZE_M 1 \
    --MUL_ROUTED_WEIGHT "True" --top_k 1 \
    --compute_type "FP16" \
    --has_zp "False" \
    --weight_bits 4 \
    --group_size 128 \
    --m 222 --n 4096 --k 7168 \
    --num_experts 256 \
    --output_dir $OUTPUT_DIR --tune

python $DIR/fused_moe_kernel_gptq_awq.py \
    --BLOCK_SIZE_M 64 --BLOCK_SIZE_N 32 --BLOCK_SIZE_K 64 --GROUP_SIZE_M 1 \
    --MUL_ROUTED_WEIGHT "True" --top_k 1 \
    --compute_type "FP16" \
    --has_zp "False" \
    --weight_bits 4 \
    --group_size 128 \
    --m 32 --n 4096 --k 7168 \
    --num_experts 256 \
    --output_dir $OUTPUT_DIR --tune

python $DIR/fused_moe_kernel_gptq_awq.py \
    --BLOCK_SIZE_M 64 --BLOCK_SIZE_N 64 --BLOCK_SIZE_K 32 --GROUP_SIZE_M 1 \
    --MUL_ROUTED_WEIGHT "True" --top_k 1 \
    --compute_type "FP16" \
    --has_zp "False" \
    --weight_bits 4 \
    --group_size 128 \
    --m 222 --n 4096 --k 7168 \
    --num_experts 256 \
    --output_dir $OUTPUT_DIR --tune

python $DIR/fused_moe_kernel_gptq_awq.py \
    --BLOCK_SIZE_M 16 --BLOCK_SIZE_N 32 --BLOCK_SIZE_K 64 --GROUP_SIZE_M 1 \
    --MUL_ROUTED_WEIGHT "False" --top_k 8 \
    --compute_type "BF16" \
    --has_zp "False" \
    --weight_bits 4 \
    --group_size 128 \
    --m 32 --n 4096 --k 7168 \
    --num_experts 256 \
    --output_dir $OUTPUT_DIR --tune

python $DIR/fused_moe_kernel_gptq_awq.py \
    --BLOCK_SIZE_M 16 --BLOCK_SIZE_N 64 --BLOCK_SIZE_K 32 --GROUP_SIZE_M 1 \
    --MUL_ROUTED_WEIGHT "False" --top_k 8 \
    --compute_type "BF16" \
    --has_zp "False" \
    --weight_bits 4 \
    --group_size 128 \
    --m 222 --n 4096 --k 7168 \
    --num_experts 256 \
    --output_dir $OUTPUT_DIR --tune

python $DIR/fused_moe_kernel_gptq_awq.py \
    --BLOCK_SIZE_M 32 --BLOCK_SIZE_N 32 --BLOCK_SIZE_K 64 --GROUP_SIZE_M 1 \
    --MUL_ROUTED_WEIGHT "False" --top_k 8 \
    --compute_type "BF16" \
    --has_zp "False" \
    --weight_bits 4 \
    --group_size 128 \
    --m 32 --n 4096 --k 7168 \
    --num_experts 256 \
    --output_dir $OUTPUT_DIR --tune

python $DIR/fused_moe_kernel_gptq_awq.py \
    --BLOCK_SIZE_M 32 --BLOCK_SIZE_N 64 --BLOCK_SIZE_K 32 --GROUP_SIZE_M 1 \
    --MUL_ROUTED_WEIGHT "False" --top_k 8 \
    --compute_type "BF16" \
    --has_zp "False" \
    --weight_bits 4 \
    --group_size 128 \
    --m 222 --n 4096 --k 7168 \
    --num_experts 256 \
    --output_dir $OUTPUT_DIR --tune

python $DIR/fused_moe_kernel_gptq_awq.py \
    --BLOCK_SIZE_M 64 --BLOCK_SIZE_N 32 --BLOCK_SIZE_K 64 --GROUP_SIZE_M 1 \
    --MUL_ROUTED_WEIGHT "False" --top_k 8 \
    --compute_type "BF16" \
    --has_zp "False" \
    --weight_bits 4 \
    --group_size 128 \
    --m 32 --n 4096 --k 7168 \
    --num_experts 256 \
    --output_dir $OUTPUT_DIR --tune

python $DIR/fused_moe_kernel_gptq_awq.py \
    --BLOCK_SIZE_M 64 --BLOCK_SIZE_N 64 --BLOCK_SIZE_K 32 --GROUP_SIZE_M 1 \
    --MUL_ROUTED_WEIGHT "False" --top_k 8 \
    --compute_type "BF16" \
    --has_zp "False" \
    --weight_bits 4 \
    --group_size 128 \
    --m 222 --n 4096 --k 7168 \
    --num_experts 256 \
    --output_dir $OUTPUT_DIR --tune

python $DIR/fused_moe_kernel_gptq_awq.py \
    --BLOCK_SIZE_M 16 --BLOCK_SIZE_N 32 --BLOCK_SIZE_K 64 --GROUP_SIZE_M 1 \
    --MUL_ROUTED_WEIGHT "True" --top_k 1 \
    --compute_type "BF16" \
    --has_zp "False" \
    --weight_bits 4 \
    --group_size 128 \
    --m 32 --n 4096 --k 7168 \
    --num_experts 256 \
    --output_dir $OUTPUT_DIR --tune

python $DIR/fused_moe_kernel_gptq_awq.py \
    --BLOCK_SIZE_M 16 --BLOCK_SIZE_N 64 --BLOCK_SIZE_K 32 --GROUP_SIZE_M 1 \
    --MUL_ROUTED_WEIGHT "True" --top_k 1 \
    --compute_type "BF16" \
    --has_zp "False" \
    --weight_bits 4 \
    --group_size 128 \
    --m 222 --n 4096 --k 7168 \
    --num_experts 256 \
    --output_dir $OUTPUT_DIR --tune

python $DIR/fused_moe_kernel_gptq_awq.py \
    --BLOCK_SIZE_M 32 --BLOCK_SIZE_N 32 --BLOCK_SIZE_K 64 --GROUP_SIZE_M 1 \
    --MUL_ROUTED_WEIGHT "True" --top_k 1 \
    --compute_type "BF16" \
    --has_zp "False" \
    --weight_bits 4 \
    --group_size 128 \
    --m 32 --n 4096 --k 7168 \
    --num_experts 256 \
    --output_dir $OUTPUT_DIR --tune

python $DIR/fused_moe_kernel_gptq_awq.py \
    --BLOCK_SIZE_M 32 --BLOCK_SIZE_N 64 --BLOCK_SIZE_K 32 --GROUP_SIZE_M 1 \
    --MUL_ROUTED_WEIGHT "True" --top_k 1 \
    --compute_type "BF16" \
    --has_zp "False" \
    --weight_bits 4 \
    --group_size 128 \
    --m 222 --n 4096 --k 7168 \
    --num_experts 256 \
    --output_dir $OUTPUT_DIR --tune

python $DIR/fused_moe_kernel_gptq_awq.py \
    --BLOCK_SIZE_M 64 --BLOCK_SIZE_N 32 --BLOCK_SIZE_K 64 --GROUP_SIZE_M 1 \
    --MUL_ROUTED_WEIGHT "True" --top_k 1 \
    --compute_type "BF16" \
    --has_zp "False" \
    --weight_bits 4 \
    --group_size 128 \
    --m 32 --n 4096 --k 7168 \
    --num_experts 256 \
    --output_dir $OUTPUT_DIR --tune

python $DIR/fused_moe_kernel_gptq_awq.py \
    --BLOCK_SIZE_M 64 --BLOCK_SIZE_N 64 --BLOCK_SIZE_K 32 --GROUP_SIZE_M 1 \
    --MUL_ROUTED_WEIGHT "True" --top_k 1 \
    --compute_type "BF16" \
    --has_zp "False" \
    --weight_bits 4 \
    --group_size 128 \
    --m 222 --n 4096 --k 7168 \
    --num_experts 256 \
    --output_dir $OUTPUT_DIR --tune