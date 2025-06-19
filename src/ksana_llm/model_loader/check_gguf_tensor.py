# Copyright 2025 Tencent Inc.  All rights reserved.
#
# ==============================================================================

import os
import sys

import torch
import numpy as np
from gguf.gguf_reader import GGUFReader


def gguf_to_torch_dtype(dtype_name: str):
    dtype_mapping = {
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "F32": torch.float32
    }
    return dtype_mapping[dtype_name]


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(1)

    model_file = sys.argv[1]
    if not os.path.isfile(model_file):
        print("Model file {} is not exist.".format(model_file))
        sys.exit(1)

    dump_path = os.getcwd() + "/" + os.path.basename(model_file)
    if not os.path.isdir(dump_path):
        print("Dump path {} is not exist.".format(dump_path))
        sys.exit(1)

    reader = GGUFReader(model_file)
    for tensor in reader.tensors:
        tensor_name = tensor.name
        tensor_data_file = dump_path + "/" + tensor_name + ".npy"

        # tensor file must be exist.
        if not os.path.isfile(tensor_data_file):
            print("Tensor file {} is not exist.".format(tensor_data_file))
            sys.exit(1)

        read_data_arr = np.load(tensor_data_file)

        # convert gguf tensor to torch.Tensor
        data_bytes = tensor.data
        torch_dtype = gguf_to_torch_dtype(tensor.tensor_type.name)
        torch_shape = tuple(tensor.shape)
        torch_tensor = torch.frombuffer(
            data_bytes,
            dtype=torch_dtype,
            count=np.prod(torch_shape)).reshape(torch_shape)

        # bf16 is not supported by numpy, cast to float32
        if torch_dtype == torch.bfloat16:
            torch_tensor = torch_tensor.to(torch.float32)

        base_data_arr = torch_tensor.numpy()
        full_matched = np.allclose(read_data_arr, base_data_arr)
        if not full_matched:
            print("Tensor {} is not matched, error.".format(tensor_name))
            sys.exit(1)
        print("Tensor {} full matched, shape: {}, dtype: {}".format(
            tensor_name, base_data_arr.shape, base_data_arr.dtype))

    # All check ok.
    print("All check passed.")
    open(dump_path + "/SUCCESS", 'a').close()
    sys.exit(0)
