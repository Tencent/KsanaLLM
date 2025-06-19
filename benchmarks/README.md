# Test Set Description
 - ShareGPT: The [data file](./share_gpt_500.csv) is pre-placed in the current directory. It contains 500 records randomly sampled (using random seed = 0) from the original [ShareGPT dataset](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered). 
    - To use this dataset for benchmarking, simply specify `--dataset-name=sharegpt500`.
    - There is **no need** to explicitly provide `--dataset_path` or `--input_csv`.
 - LongBench V2: The data file should be downloaded manually from [HuggingFace](https://huggingface.co/datasets/THUDM/LongBench-v2/resolve/2b48e49/data.json) before benchmarking. It contains 503 challenging multiple-choice questions with context lengths ranging from 8k to 2M words. 
    - To use this dataset for benchmarking, you need to specify the path to the data file using `--dataset_path`.
    - The dataset supports two prompt settings: Specify `--dataset-name=longbenchV2withCtx` to **include the full background context** in each prompt; Specify `--dataset-name=longbenchV2noCtx` to exclude the context from prompts.
    - When starting the inference server, try to increase `--max-model-len` (if using vLLM) or `max_token_len` (if using KsanaLLM)

# Download model
```
cd ${GIT_PROJECT_REPO_ROOT}/src/ksana_llm/python

# download huggingface model for example:
# Note: Make sure git-lfs is installed.
git clone https://huggingface.co/NousResearch/Llama-2-7b-hf

```

# Ksana
## Start server
```
cd ${GIT_PROJECT_REPO_ROOT}/src/ksana_llm/python

export CUDA_VISIBLE_DEVICES=xx

# launch server
python serving_server.py \
    --config_file ../../../examples/ksana_llm2-7b.yaml \
    --port 8080
```
Change config file when trying other options

## Start benchmark
```
cd ${GIT_PROJECT_REPO_ROOT}/benchmarks

# benchmark
python benchmark_throughput.py --input_csv benchmark_input.csv --port 8080 \
    --backend ksana --model_type llama \
    --perf_csv ksana_perf.csv > ksana_stdout.txt 2>&1

# benchmark triton_backend with grpc streaming
python benchmark_throughput.py --host localhost \
    --port 8080 \
    --input_csv benchmark_input.csv  \
    --perf_csv ksana_perf.csv \
    --backend triton-grpc \
    --triton_model_name ksana_llm \
    --tokenizer_path /model_path/ \
    --stream
```

# vLLM
## Start server
```
export MODEL_PATH=${GIT_PROJECT_REPO_ROOT}/src/ksana_llm/python/Llama-2-7b-hf
export CUDA_VISIBLE_DEVICES=xx

python -m vllm.entrypoints.api_server \
     --model $MODEL_PATH \
     --tokenizer $MODEL_PATH \
     --trust-remote-code \
     --max-model-len 1536 \
     --pipeline-parallel-size 1 \
     --tensor-parallel-size 1 \
     --gpu-memory-utilization 0.94 \
     --disable-log-requests \
     --port 8080 
```

## Start benchmark
```
python benchmark_throughput.py --port 8080  --input_csv benchmark_input.csv  \
    --model_type llama \
    --tokenizer_path $MODEL_PATH  \
    --backend vllm \
    --perf_csv vllm_perf.csv > vllm_stdout.txt 2>&1
```
