#!/bin/bash
DIR=$1
OUTPUT_DIR=$2

#   BFP16
python $DIR/fused_moe.py \
    --output_dir $OUTPUT_DIR \
    --group_n 0 \
    --group_k 0 \
    --BLOCK_SIZE_M 64 \
    --BLOCK_SIZE_N 64 \
    --BLOCK_SIZE_K 32 \
    --GROUP_SIZE_M 8 \
    --MUL_ROUTED_WEIGHT "False" \
    --top_k 8 \
    --compute_type "BF16" \
    --use_fp8_w8a8 "False" \
    --use_int8_w8a16 "False" --tune

python $DIR/fused_moe.py \
    --output_dir $OUTPUT_DIR \
    --group_n 0 \
    --group_k 0 \
    --BLOCK_SIZE_M 16 \
    --BLOCK_SIZE_N 32 \
    --BLOCK_SIZE_K 64 \
    --GROUP_SIZE_M 1 \
    --MUL_ROUTED_WEIGHT "False" \
    --top_k 8 \
    --compute_type "BF16" \
    --use_fp8_w8a8 "False" \
    --use_int8_w8a16 "False" \
    --m 300 --tune

python $DIR/fused_moe.py \
    --output_dir $OUTPUT_DIR \
    --group_n 0 \
    --group_k 0 \
    --BLOCK_SIZE_M 64 \
    --BLOCK_SIZE_N 64 \
    --BLOCK_SIZE_K 32 \
    --GROUP_SIZE_M 8 \
    --MUL_ROUTED_WEIGHT "False" \
    --top_k 6 \
    --compute_type "BF16" \
    --use_fp8_w8a8 "False" \
    --use_int8_w8a16 "False" --tune

python $DIR/fused_moe.py \
    --output_dir $OUTPUT_DIR \
    --group_n 0 \
    --group_k 0 \
    --BLOCK_SIZE_M 16 \
    --BLOCK_SIZE_N 32 \
    --BLOCK_SIZE_K 64 \
    --GROUP_SIZE_M 1 \
    --MUL_ROUTED_WEIGHT "False" \
    --top_k 6 \
    --compute_type "BF16" \
    --use_fp8_w8a8 "False" \
    --use_int8_w8a16 "False" \
    --m 300 --tune

python $DIR/fused_moe.py \
    --output_dir $OUTPUT_DIR \
    --group_n 0 \
    --group_k 0 \
    --BLOCK_SIZE_M 64 \
    --BLOCK_SIZE_N 64 \
    --BLOCK_SIZE_K 32 \
    --GROUP_SIZE_M 8 \
    --MUL_ROUTED_WEIGHT "True" \
    --top_k 1 \
    --compute_type "BF16" \
    --use_fp8_w8a8 "False" \
    --use_int8_w8a16 "False" \
    --n 2048 --tune

python $DIR/fused_moe.py \
    --output_dir $OUTPUT_DIR \
    --group_n 0 \
    --group_k 0 \
    --BLOCK_SIZE_M 16 \
    --BLOCK_SIZE_N 32 \
    --BLOCK_SIZE_K 64 \
    --GROUP_SIZE_M 1 \
    --MUL_ROUTED_WEIGHT "True" \
    --top_k 1 \
    --compute_type "BF16" \
    --use_fp8_w8a8 "False" \
    --use_int8_w8a16 "False" \
    --m 300 --tune

#   FP16
python $DIR/fused_moe.py \
    --output_dir $OUTPUT_DIR \
    --group_n 0 \
    --group_k 0 \
    --BLOCK_SIZE_M 64 \
    --BLOCK_SIZE_N 64 \
    --BLOCK_SIZE_K 32 \
    --GROUP_SIZE_M 8 \
    --MUL_ROUTED_WEIGHT "False" \
    --top_k 8 \
    --compute_type "FP16" \
    --use_fp8_w8a8 "False" \
    --use_int8_w8a16 "False" --tune

python $DIR/fused_moe.py \
    --output_dir $OUTPUT_DIR \
    --group_n 0 \
    --group_k 0 \
    --BLOCK_SIZE_M 16 \
    --BLOCK_SIZE_N 32 \
    --BLOCK_SIZE_K 64 \
    --GROUP_SIZE_M 1 \
    --MUL_ROUTED_WEIGHT "False" \
    --top_k 8 \
    --compute_type "FP16" \
    --use_fp8_w8a8 "False" \
    --use_int8_w8a16 "False" \
    --m 300 --tune

python $DIR/fused_moe.py \
    --output_dir $OUTPUT_DIR \
    --group_n 0 \
    --group_k 0 \
    --BLOCK_SIZE_M 64 \
    --BLOCK_SIZE_N 64 \
    --BLOCK_SIZE_K 32 \
    --GROUP_SIZE_M 8 \
    --MUL_ROUTED_WEIGHT "False" \
    --top_k 6 \
    --compute_type "FP16" \
    --use_fp8_w8a8 "False" \
    --use_int8_w8a16 "False" --tune

python $DIR/fused_moe.py \
    --output_dir $OUTPUT_DIR \
    --group_n 0 \
    --group_k 0 \
    --BLOCK_SIZE_M 16 \
    --BLOCK_SIZE_N 32 \
    --BLOCK_SIZE_K 64 \
    --GROUP_SIZE_M 1 \
    --MUL_ROUTED_WEIGHT "False" \
    --top_k 6 \
    --compute_type "FP16" \
    --use_fp8_w8a8 "False" \
    --use_int8_w8a16 "False" \
    --m 300 --tune

python $DIR/fused_moe.py \
    --output_dir $OUTPUT_DIR \
    --group_n 0 \
    --group_k 0 \
    --BLOCK_SIZE_M 64 \
    --BLOCK_SIZE_N 64 \
    --BLOCK_SIZE_K 32 \
    --GROUP_SIZE_M 8 \
    --MUL_ROUTED_WEIGHT "True" \
    --top_k 1 \
    --compute_type "FP16" \
    --use_fp8_w8a8 "False" \
    --use_int8_w8a16 "False" \
    --n 2048 --tune

python $DIR/fused_moe.py \
    --output_dir $OUTPUT_DIR \
    --group_n 0 \
    --group_k 0 \
    --BLOCK_SIZE_M 16 \
    --BLOCK_SIZE_N 32 \
    --BLOCK_SIZE_K 64 \
    --GROUP_SIZE_M 1 \
    --MUL_ROUTED_WEIGHT "True" \
    --top_k 1 \
    --compute_type "FP16" \
    --use_fp8_w8a8 "False" \
    --use_int8_w8a16 "False" \
    --m 300 --tune

# Temporary environment variable to turn off even_ks.
if [ -n "$DISABLE_EVEN_KS" ] && [ "$DISABLE_EVEN_KS" -eq 1 ]
then
  echo "even_Ks is disabled"
  even_Ks="False"  
else
  echo "even_Ks is enabled"
  even_Ks="True"
fi

#   FP8-BF16
python $DIR/fused_moe.py \
    --output_dir $OUTPUT_DIR \
    --group_n 128 \
    --group_k 128 \
    --BLOCK_SIZE_M 64 \
    --BLOCK_SIZE_N 128 \
    --BLOCK_SIZE_K 128 \
    --GROUP_SIZE_M 32 \
    --num_warps 4 \
    --num_stages 3 \
    --MUL_ROUTED_WEIGHT "False" \
    --top_k 8 \
    --compute_type "BF16" \
    --use_fp8_w8a8 "True" \
    --use_int8_w8a16 "False" \
    --even_Ks "${even_Ks}"

python $DIR/fused_moe.py \
    --output_dir $OUTPUT_DIR \
    --group_n 128 \
    --group_k 128 \
    --BLOCK_SIZE_M 64 \
    --BLOCK_SIZE_N 128 \
    --BLOCK_SIZE_K 128 \
    --GROUP_SIZE_M 1 \
    --num_warps 4 \
    --num_stages 3 \
    --MUL_ROUTED_WEIGHT "False" \
    --top_k 8 \
    --compute_type "BF16" \
    --use_fp8_w8a8 "True" \
    --use_int8_w8a16 "False" \
    --even_Ks "${even_Ks}"

python $DIR/fused_moe.py \
    --output_dir $OUTPUT_DIR \
    --group_n 128 \
    --group_k 128 \
    --BLOCK_SIZE_M 16 \
    --BLOCK_SIZE_N 128 \
    --BLOCK_SIZE_K 128 \
    --GROUP_SIZE_M 32 \
    --num_warps 4 \
    --num_stages 3 \
    --MUL_ROUTED_WEIGHT "False" \
    --top_k 8 \
    --compute_type "BF16" \
    --use_fp8_w8a8 "True" \
    --use_int8_w8a16 "False" \
    --even_Ks "${even_Ks}"


python $DIR/fused_moe.py \
    --output_dir $OUTPUT_DIR \
    --group_n 128 \
    --group_k 128 \
    --BLOCK_SIZE_M 64 \
    --BLOCK_SIZE_N 128 \
    --BLOCK_SIZE_K 128 \
    --GROUP_SIZE_M 32 \
    --MUL_ROUTED_WEIGHT "False" \
    --top_k 6 \
    --compute_type "BF16" \
    --use_fp8_w8a8 "True" \
    --use_int8_w8a16 "False" --tune

python $DIR/fused_moe.py \
    --output_dir $OUTPUT_DIR \
    --group_n 128 \
    --group_k 128 \
    --BLOCK_SIZE_M 64 \
    --BLOCK_SIZE_N 128 \
    --BLOCK_SIZE_K 128 \
    --GROUP_SIZE_M 1 \
    --MUL_ROUTED_WEIGHT "False" \
    --top_k 6 \
    --compute_type "BF16" \
    --use_fp8_w8a8 "True" \
    --use_int8_w8a16 "False" --tune

python $DIR/fused_moe.py \
    --output_dir $OUTPUT_DIR \
    --group_n 128 \
    --group_k 128 \
    --BLOCK_SIZE_M 64 \
    --BLOCK_SIZE_N 128 \
    --BLOCK_SIZE_K 128 \
    --GROUP_SIZE_M 32 \
    --num_warps 4 \
    --num_stages 3 \
    --MUL_ROUTED_WEIGHT "True" \
    --top_k 1 \
    --compute_type "BF16" \
    --use_fp8_w8a8 "True" \
    --use_int8_w8a16 "False" \
    --even_Ks "${even_Ks}"

python $DIR/fused_moe.py \
    --output_dir $OUTPUT_DIR \
    --group_n 128 \
    --group_k 128 \
    --BLOCK_SIZE_M 64 \
    --BLOCK_SIZE_N 128 \
    --BLOCK_SIZE_K 128 \
    --GROUP_SIZE_M 1 \
    --num_warps 4 \
    --num_stages 3 \
    --MUL_ROUTED_WEIGHT "True" \
    --top_k 1 \
    --compute_type "BF16" \
    --use_fp8_w8a8 "True" \
    --use_int8_w8a16 "False" \
    --even_Ks "${even_Ks}"

python $DIR/fused_moe.py \
    --output_dir $OUTPUT_DIR \
    --group_n 128 \
    --group_k 128 \
    --BLOCK_SIZE_M 16 \
    --BLOCK_SIZE_N 128 \
    --BLOCK_SIZE_K 128 \
    --GROUP_SIZE_M 32 \
    --num_warps 4 \
    --num_stages 3 \
    --MUL_ROUTED_WEIGHT "True" \
    --top_k 1 \
    --compute_type "BF16" \
    --use_fp8_w8a8 "True" \
    --use_int8_w8a16 "False" \
    --even_Ks "${even_Ks}"


#   FP8-FP16
python $DIR/fused_moe.py \
    --output_dir $OUTPUT_DIR \
    --group_n 128 \
    --group_k 128 \
    --BLOCK_SIZE_M 64 \
    --BLOCK_SIZE_N 128 \
    --BLOCK_SIZE_K 128 \
    --GROUP_SIZE_M 32 \
    --num_warps 4 \
    --num_stages 3 \
    --MUL_ROUTED_WEIGHT "False" \
    --top_k 8 \
    --compute_type "FP16" \
    --use_fp8_w8a8 "True" \
    --use_int8_w8a16 "False" \
    --even_Ks "${even_Ks}"


python $DIR/fused_moe.py \
    --output_dir $OUTPUT_DIR \
    --group_n 128 \
    --group_k 128 \
    --BLOCK_SIZE_M 64 \
    --BLOCK_SIZE_N 128 \
    --BLOCK_SIZE_K 128 \
    --GROUP_SIZE_M 1 \
    --num_warps 4 \
    --num_stages 3 \
    --MUL_ROUTED_WEIGHT "False" \
    --top_k 8 \
    --compute_type "FP16" \
    --use_fp8_w8a8 "True" \
    --use_int8_w8a16 "False" \
    --even_Ks "${even_Ks}"

python $DIR/fused_moe.py \
    --output_dir $OUTPUT_DIR \
    --group_n 128 \
    --group_k 128 \
    --BLOCK_SIZE_M 16 \
    --BLOCK_SIZE_N 128 \
    --BLOCK_SIZE_K 128 \
    --GROUP_SIZE_M 32 \
    --num_warps 4 \
    --num_stages 3 \
    --MUL_ROUTED_WEIGHT "False" \
    --top_k 8 \
    --compute_type "FP16" \
    --use_fp8_w8a8 "True" \
    --use_int8_w8a16 "False" \
    --even_Ks "${even_Ks}"


python $DIR/fused_moe.py \
    --output_dir $OUTPUT_DIR \
    --group_n 128 \
    --group_k 128 \
    --BLOCK_SIZE_M 64 \
    --BLOCK_SIZE_N 128 \
    --BLOCK_SIZE_K 128 \
    --GROUP_SIZE_M 32 \
    --MUL_ROUTED_WEIGHT "False" \
    --top_k 6 \
    --compute_type "FP16" \
    --use_fp8_w8a8 "True" \
    --use_int8_w8a16 "False" --tune


python $DIR/fused_moe.py \
    --output_dir $OUTPUT_DIR \
    --group_n 128 \
    --group_k 128 \
    --BLOCK_SIZE_M 64 \
    --BLOCK_SIZE_N 128 \
    --BLOCK_SIZE_K 128 \
    --GROUP_SIZE_M 1 \
    --MUL_ROUTED_WEIGHT "False" \
    --top_k 6 \
    --compute_type "FP16" \
    --use_fp8_w8a8 "True" \
    --use_int8_w8a16 "False" --tune


python $DIR/fused_moe.py \
    --output_dir $OUTPUT_DIR \
    --group_n 128 \
    --group_k 128 \
    --BLOCK_SIZE_M 64 \
    --BLOCK_SIZE_N 128 \
    --BLOCK_SIZE_K 128 \
    --GROUP_SIZE_M 32 \
    --num_warps 4 \
    --num_stages 3 \
    --MUL_ROUTED_WEIGHT "True" \
    --top_k 1 \
    --compute_type "FP16" \
    --use_fp8_w8a8 "True" \
    --use_int8_w8a16 "False" \
    --even_Ks "${even_Ks}"

python $DIR/fused_moe.py \
    --output_dir $OUTPUT_DIR \
    --group_n 128 \
    --group_k 128 \
    --BLOCK_SIZE_M 64 \
    --BLOCK_SIZE_N 128 \
    --BLOCK_SIZE_K 128 \
    --GROUP_SIZE_M 1 \
    --MUL_ROUTED_WEIGHT "True" \
    --top_k 1 \
    --num_warps 4 \
    --num_stages 3 \
    --compute_type "FP16" \
    --use_fp8_w8a8 "True" \
    --use_int8_w8a16 "False" \
    --even_Ks "${even_Ks}"

python $DIR/fused_moe.py \
    --output_dir $OUTPUT_DIR \
    --group_n 128 \
    --group_k 128 \
    --BLOCK_SIZE_M 16 \
    --BLOCK_SIZE_N 128 \
    --BLOCK_SIZE_K 128 \
    --GROUP_SIZE_M 32 \
    --num_warps 4 \
    --num_stages 3 \
    --MUL_ROUTED_WEIGHT "True" \
    --top_k 1 \
    --compute_type "FP16" \
    --use_fp8_w8a8 "True" \
    --use_int8_w8a16 "False" \
    --even_Ks "${even_Ks}" \