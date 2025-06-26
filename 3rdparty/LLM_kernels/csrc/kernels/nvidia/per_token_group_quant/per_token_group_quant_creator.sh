#!/bin/bash

# only use one GPU
if [[ -v CUDA_VISIBLE_DEVICES ]]; then
    IFS=',' read -ra devices <<< "$CUDA_VISIBLE_DEVICES"
    export CUDA_VISIBLE_DEVICES=${devices[0]}
fi

DIR=$1
OUTPUT_DIR=$2

python $DIR/per_token_group_quant_fp8.py \
    --output_dir $OUTPUT_DIR \
    --compute_type "FP16" \
    --colmajor "True"

python $DIR/per_token_group_quant_fp8.py \
    --output_dir $OUTPUT_DIR \
    --compute_type "FP16" \
    --colmajor "False"

python $DIR/per_token_group_quant_fp8.py \
    --output_dir $OUTPUT_DIR \
    --compute_type "BF16" \
    --colmajor "True"

python $DIR/per_token_group_quant_fp8.py \
    --output_dir $OUTPUT_DIR \
    --compute_type "BF16" \
    --colmajor "False"
