# GEMM Algorithm Search Usage Guide

1. Record the M, N, and K values required for GEMM computations during the inference process (it is recommended to use tools such as NVTX or Nsight System), and fill in the `gemm_problem_space_template.csv` according to the template in this directory.

2. Generate the GEMM optimization configuration file `gemm_algo_map_demo.yaml` using the command:  
   `python gemm_algo_config_generator.py --input_file gemm_problem_space_template.csv --output_file gemm_algo_map_demo.yaml`.

3. Place the GEMM optimization configuration file `gemm_algo_map_demo.yaml` at the same level as `config.json` in the model directory.

4. The optimization search will record performance data during the process. For example, `gpu_elapsed_ms` will be recorded under each algorithm in `gemm_algo_map_demo.yaml`.