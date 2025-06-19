#!/bin/bash

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
