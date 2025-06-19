#!/bin/bash

DIR=$1
OUTPUT_DIR=$2

python $DIR/decode_softmax_reducev_fwd.py \
    --output_dir $OUTPUT_DIR \
    --input_type "FP16"

python $DIR/decode_softmax_reducev_fwd.py \
    --output_dir $OUTPUT_DIR \
    --input_type "BF16"