# Copyright 2024 Tencent Inc.  All rights reserved.
# ==============================================================================

import argparse
import csv
import deep_gemm_kernel_generator
import os
import logging
import subprocess

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def check_cublas_algo_searcher_avaliable(path: str):
    return os.path.exists(path)


# Define a function to parse command line arguments
def args_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generator_bin_path',
                        type=str,
                        default="../../../../build/bin/search_best_gemm_algo",
                        help='The gemm algo searcher binary file path')
    parser.add_argument('--input_file',
                        type=str,
                        default="gemm_problem_space_template.csv",
                        help='GEMM problem space')
    parser.add_argument('--output_file',
                        type=str,
                        default="gemm_algo_map.yaml",
                        help='profile result')
    args = parser.parse_args()
    return args


def read_csv_file(file_path):
    data = []
    with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            data.append(row)
    return data


if __name__ == "__main__":
    args = args_config()
    csv_file_path = args.input_file
    # setting DeepGEMM kernel generation dir
    os.environ['DG_CACHE_DIR'] = os.getcwd()

    is_cublas_algo_searcher_avaliable = check_cublas_algo_searcher_avaliable(
        args.generator_bin_path)

    csv_data = read_csv_file(csv_file_path)
    for row in csv_data:
        batch_size = row['batch_size']
        m = int(row['m'])
        n = int(row['n'])
        k = int(row['k'])
        input_dtype = row['input_dtype']
        output_dtype = row['output_dtype']
        inner_compute_dtype = row['inner_compute_dtype']
        input_a_transop = row['input_a_transop']
        input_b_transop = row['input_b_transop']
        op_type = int(row['op_type'])

        # NOTE(karlluo): DeepGEMM only support Hopper, check first
        if deep_gemm_kernel_generator.is_deep_gemm_available:
            logging.info(f"Generating DeepGEMM on shape m,n,k = {m},{n},{k}")
            deep_gemm_kernel_generator.generate_deep_gemm_kernel(
                m, n, k, input_dtype, output_dtype, op_type, args.output_file)

        if is_cublas_algo_searcher_avaliable and op_type == 0:
            logging.info(f"Generating cuBlas on shape m,n,k = {m},{n},{k}")
            command = [args.generator_bin_path]
            for metric_key in row:
                command += ["--%s" % metric_key, row[metric_key]]
            print(" ".join(command))
            return_code = subprocess.check_call(command)
            assert return_code == 0, f'Failed to run cuBlas searcher with command {" ".join(command)}'
