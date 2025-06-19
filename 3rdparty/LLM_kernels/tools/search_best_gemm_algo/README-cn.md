# GEMM算法搜索使用指南

1. 将推理过程中需要的GEMM计算中M，N，K值记录下来（建议使用NVTX或Nsight System等工具），按照本目录的gemm_problem_space_template.csv填写。

2. 使用`python gemm_algo_config_generator.py --input_file gemm_problem_space_template.csv --output_file gemm_algo_map_demo.yaml`生成GEMM优化配置文件gemm_algo_map_demo.yaml。

3. 将GEMM优化配置文件gemm_algo_map_demo.yaml放到模型目录下与config.json同级。

4. 搜索优化会记录优化过程中的性能数据，在gemm_algo_map_demo.yaml每个algo下例如：gpu_elapsed_ms。