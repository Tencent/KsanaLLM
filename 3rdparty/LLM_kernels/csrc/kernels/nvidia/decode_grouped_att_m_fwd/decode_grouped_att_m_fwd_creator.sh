#!/bin/bash

# only use one GPU
if [[ -v CUDA_VISIBLE_DEVICES ]]; then
    IFS=',' read -ra devices <<< "$CUDA_VISIBLE_DEVICES"
    export CUDA_VISIBLE_DEVICES=${devices[0]}
fi

DIR=$1
OUTPUT_DIR=$2

python $DIR/decode_grouped_att_m_fwd.py \
    --output_dir $OUTPUT_DIR \
    --q_head_num 128 \
    --kv_group_num 128 \
    --input_type "BF16"
python $DIR/decode_grouped_att_m_fwd.py \
    --output_dir $OUTPUT_DIR \
    --q_head_num 64 \
    --kv_group_num 64 \
    --input_type "BF16"
python $DIR/decode_grouped_att_m_fwd.py \
    --output_dir $OUTPUT_DIR \
    --q_head_num 32 \
    --kv_group_num 32 \
    --input_type "BF16"
python $DIR/decode_grouped_att_m_fwd.py \
    --output_dir $OUTPUT_DIR \
    --q_head_num 16 \
    --kv_group_num 16 \
    --input_type "BF16"
python $DIR/decode_grouped_att_m_fwd.py \
    --output_dir $OUTPUT_DIR \
    --q_head_num 8 \
    --kv_group_num 8 \
    --input_type "BF16"
python $DIR/decode_grouped_att_m_fwd.py \
    --output_dir $OUTPUT_DIR \
    --q_head_num 4 \
    --kv_group_num 4 \
    --input_type "BF16"
python $DIR/decode_grouped_att_m_fwd.py \
    --output_dir $OUTPUT_DIR \
    --q_head_num 2 \
    --kv_group_num 2 \
    --input_type "BF16"
python $DIR/decode_grouped_att_m_fwd.py \
    --output_dir $OUTPUT_DIR \
    --q_head_num 1 \
    --kv_group_num 1 \
    --input_type "BF16"

python $DIR/decode_grouped_att_m_fwd.py \
    --output_dir $OUTPUT_DIR \
    --q_head_num 128 \
    --kv_group_num 128 \
    --input_type "FP16"
python $DIR/decode_grouped_att_m_fwd.py \
    --output_dir $OUTPUT_DIR \
    --q_head_num 64 \
    --kv_group_num 64 \
    --input_type "FP16"
python $DIR/decode_grouped_att_m_fwd.py \
    --output_dir $OUTPUT_DIR \
    --q_head_num 32 \
    --kv_group_num 32 \
    --input_type "FP16"
python $DIR/decode_grouped_att_m_fwd.py \
    --output_dir $OUTPUT_DIR \
    --q_head_num 16 \
    --kv_group_num 16 \
    --input_type "FP16"
python $DIR/decode_grouped_att_m_fwd.py \
    --output_dir $OUTPUT_DIR \
    --q_head_num 8 \
    --kv_group_num 8 \
    --input_type "FP16"
python $DIR/decode_grouped_att_m_fwd.py \
    --output_dir $OUTPUT_DIR \
    --q_head_num 4 \
    --kv_group_num 4 \
    --input_type "FP16"
python $DIR/decode_grouped_att_m_fwd.py \
    --output_dir $OUTPUT_DIR \
    --q_head_num 2 \
    --kv_group_num 2 \
    --input_type "FP16"
python $DIR/decode_grouped_att_m_fwd.py \
    --output_dir $OUTPUT_DIR \
    --q_head_num 1 \
    --kv_group_num 1 \
    --input_type "FP16"