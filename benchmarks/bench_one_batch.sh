# run one batch with specified batch_size、 input_len and output_len
BATCH_SIZE=32
INPUT_LEN=256
OUTPUT_LEN=128
python benchmark_throughput.py \
    --port=8087 \
    --warmup_num_iters 1 \
    --prompt_num 1 \
    --num_return_sequences ${BATCH_SIZE} \
    --random_input_len ${INPUT_LEN} \
    --max_new_tokens ${OUTPUT_LEN} \
    --ignore_eos \
    --tokenizer_path /dockerdata/models/DeepSeek-R1
