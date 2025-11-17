# Benchmarking Tool

This tool provides comprehensive benchmarking capabilities for various LLM serving backends including KsanaLLM, vLLM, TensorRT-LLM, and others.

## Features

- **Performance Benchmarking**: Measure throughput, latency, TTFT (Time To First Token), and other metrics
- **Multiple Backends Support**: Compatible with KsanaLLM, vLLM, TensorRT-LLM, SGLang, and more
- **Automatic Diff Checking**: Compare outputs between different benchmark runs to detect consistency issues

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

# benchmark with automatic diff checking between runs
python benchmark_throughput.py --input_csv benchmark_input.csv --port 8080 \
    --backend ksana --model_type llama \
    --perf_csv ksana_perf.csv \
    --enable_diff_check \
    --repeat_num_iters 2 \
    --diff_rouge_threshold 0.5 \
    --diff_mismatch_threshold 10 \
    --diff_output_file comparison_results.txt
```

### Diff Check Parameters
- `--enable_diff_check`: Enable automatic diff checking functionality to compare outputs between two benchmark runs
- `--diff_rouge_threshold`: ROUGE-W threshold below which detailed results are printed (default: 0.5)
- `--diff_mismatch_threshold`: First mismatch position threshold below which detailed results are printed (optional)
- `--diff_output_file`: Output file path for diff results (default: comparison_results.txt)

**Note**: When `--enable_diff_check` is enabled and `--repeat_num_iters` is less than 2, the system will automatically set it to 2 to ensure sufficient runs for comparison.

## Automatic Diff Checking

The diff checking functionality compares output consistency of the same model across multiple runs, helping detect model inference stability and reproducibility issues.

### How It Works

1. **Automatic Run Comparison**: When `--enable_diff_check` is enabled, the tool automatically runs at least two benchmark iterations
2. **Multi-dimensional Comparison**: Uses multiple metrics to evaluate text similarity:
   - **ROUGE-W F1 Score**: Word-level similarity assessment (0-1 range, higher is better)
   - **Levenshtein Distance**: Character-level edit distance (0-1 range, higher is better)
   - **First Mismatch Position**: Detects where text starts to differ (lower indicates earlier divergence)

3. **Smart Filtering**: Only text pairs meeting any of the following conditions are output in detail:
   - ROUGE-W score below the set threshold
   - First mismatch position within the set threshold range

### Use Cases

- **Model Stability Testing**: Verify if the model produces consistent outputs for the same inputs
- **Configuration Change Validation**: Compare model output differences under different configurations
- **Version Regression Testing**: Detect output changes after model updates
- **Randomness Analysis**: Evaluate the degree of randomness in model outputs

### Interpreting Results

The diff check generates a report containing:
- Detailed text comparisons (only showing samples with significant differences)
- Overall statistics (average ROUGE-W score, average edit distance, etc.)
- Trigger condition explanations (which samples were flagged and why)

### Standalone Usage of check_diff.py

You can also use the diff checking tool independently to compare any two CSV files containing text outputs:

```bash
cd ${GIT_PROJECT_REPO_ROOT}/tools/inference_diff_checker

# Basic comparison with default settings
python check_diff.py baseline_results.csv comparison_results.csv

# Show only texts with very low ROUGE-W scores
python check_diff.py baseline_results.csv comparison_results.csv --rouge-threshold 0.2

# Show texts with early mismatches (within first 5 characters)
python check_diff.py baseline_results.csv comparison_results.csv --first-mismatch-threshold 5

# Combine both thresholds and specify custom output file
python check_diff.py baseline_results.csv comparison_results.csv \
    --rouge-threshold 0.4 \
    --first-mismatch-threshold 10 \
    --output detailed_analysis.txt
```

#### CSV File Format

The CSV files should contain text data in the first column:

```csv
"This is the expected output"
"Another reference text"
"Final example text"
```

#### Command Line Parameters

- `csv_file1`: Path to the first CSV file (reference/baseline texts)
- `csv_file2`: Path to the second CSV file (comparison texts)
- `--rouge-threshold`: ROUGE-W threshold below which detailed results are printed (default: 0.5)
- `--first-mismatch-threshold`: First mismatch position threshold below which detailed results are printed (optional)
- `--output` / `-o`: Output file path for results (default: comparison_results.txt)

#### Metrics Explanation

- **ROUGE-W F1 Score**: Higher is better (0-1 range), measures word-level similarity
- **Levenshtein Ratio**: Higher is better (0-1 range), measures character-level similarity
- **First Mismatch Position**: Lower indicates earlier divergence (1-indexed)

Texts are flagged for detailed output if they meet ANY of the threshold conditions.

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
