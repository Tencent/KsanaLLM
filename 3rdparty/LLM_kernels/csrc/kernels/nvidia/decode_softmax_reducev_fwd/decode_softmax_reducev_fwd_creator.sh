#!/bin/bash

# only use one GPU
if [[ -v CUDA_VISIBLE_DEVICES ]]; then
    IFS=',' read -ra devices <<< "$CUDA_VISIBLE_DEVICES"
    export CUDA_VISIBLE_DEVICES=${devices[0]}
fi

DIR=$1
OUTPUT_DIR=$2

python $DIR/decode_softmax_reducev_fwd.py \
    --output_dir $OUTPUT_DIR \
    --input_type "FP16"

python $DIR/decode_softmax_reducev_fwd.py \
    --output_dir $OUTPUT_DIR \
    --input_type "BF16"