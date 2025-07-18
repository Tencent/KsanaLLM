#!/usr/bin/env python3

import argparse
import base64
import threading
import time

import msgpack
import numpy as np
import tritonclient.grpc as grpcclient
from LLMTextHandler.tritonft.utils import prepare_tensor


def np_to_triton_dtype(np_dtype):
    """Convert numpy data type to Triton data type"""
    if np_dtype == np.bool_:
        return "BOOL"
    elif np_dtype == np.int8:
        return "INT8"
    elif np_dtype == np.int16:
        return "INT16"
    elif np_dtype == np.int32:
        return "INT32"
    elif np_dtype == np.int64:
        return "INT64"
    elif np_dtype == np.uint8:
        return "UINT8"
    elif np_dtype == np.uint16:
        return "UINT16"
    elif np_dtype == np.uint32:
        return "UINT32"
    elif np_dtype == np.uint64:
        return "UINT64"
    elif np_dtype == np.float16:
        return "FP16"
    elif np_dtype == np.float32:
        return "FP32"
    elif np_dtype == np.float64:
        return "FP64"
    elif np_dtype == np.object_ or np_dtype == np.dtype('O'):
        return "BYTES"
    return None


def python_tensor_to_numpy(python_tensor):
    """
    Convert custom PythonTensor object to NumPy array
    """
    print(
        f"Parsing tensor: {python_tensor.keys() if isinstance(python_tensor, dict) else 'non-dict type'}")

    # Extract data, shape and dtype
    data, shape, dtype = python_tensor.get("data"), python_tensor.get(
        "shape"), python_tensor.get("dtype")

    if data is None or shape is None or dtype is None:
        print(
            f"Tensor missing required fields - "
            f"data: {'exists' if data else 'missing'}, "
            f"shape: {'exists' if shape else 'missing'}, "
            f"dtype: {'exists' if dtype else 'missing'}")
        return np.array([])  # Return empty array

    print(
        f"Tensor info - type: {dtype}, shape: {shape}, "
        f"data length: {len(data) if isinstance(data, str) else 'non-string type'}")

    # Map string data type to corresponding NumPy data type
    if dtype == "float32":
        np_dtype = np.float32
    elif dtype == "float16":
        np_dtype = np.float16
    elif dtype == "bfloat16":
        np_dtype = np.uint16
    elif dtype == "int32":
        np_dtype = np.int32
    else:
        raise ValueError(f"Unsupported data type: {dtype}")

    try:
        # Create NumPy array from raw data buffer
        decoded_data = base64.b64decode(data)
        print(f"Decoded data size: {len(decoded_data)} bytes")
        data_array = np.frombuffer(decoded_data, dtype=np_dtype)

        # Check shape compatibility
        expected_elements = np.prod(shape)
        actual_elements = data_array.size
        print(
            f"Expected elements: {expected_elements}, actual elements: {actual_elements}")

        if expected_elements != actual_elements:
            print(
                f"Warning: Shape mismatch! Expected {expected_elements} elements, got {actual_elements}")
            # If shape doesn't match, return flat array
            return data_array

        # Reshape NumPy array according to the specified shape
        numpy_array = data_array.reshape(shape)

        # Special handling for 'bfloat16' data type, convert uint16 to float32
        if dtype == "bfloat16":
            numpy_array = numpy_array.astype(np.uint32) << 16
            numpy_array = numpy_array.view(np.float32)

        return numpy_array
    except (ValueError, TypeError) as e:
        print(f"Tensor parsing error (data format): {e}")
        return np.array([])  # Return empty array on error


def show_response(result):
    """Display response results"""
    print(f"result: {result}")
    if isinstance(result, dict) and "responses" in result:
        for batch_result in result["responses"]:
            if isinstance(batch_result, dict) and "response" in batch_result:
                input_token_ids = batch_result["input_token_ids"]
                for response in batch_result["response"]:
                    target = response["target_name"]
                    python_tensor = response["tensor"]
                    print(
                        f"input_token_ids: {input_token_ids}, target: {target}, "
                        f"tensor: \n{python_tensor_to_numpy(python_tensor)}"
                    )
        print(f"message: \"{result['message']}\", code: {result['code']}")
    else:
        print(result)


def triton_forward(
    url,
    model_name,
    prompt,
    request_id="",
    target_name="layernorm",
    token_id=None,
    token_reduce_mode="GATHER_ALL",
    slice_pos=None
):
    """Send forward request to Triton server using gRPC protocol"""

    with grpcclient.InferenceServerClient(url=url, verbose=False) as client:
        done_event = threading.Event()
        # Prepare forward request data
        request_target = {
            "target_name": target_name,
            "token_reduce_mode": token_reduce_mode,
        }

        # Add token_id or slice_pos based on provided parameters
        if token_id:
            request_target["token_id"] = token_id
        if slice_pos:
            request_target["slice_pos"] = slice_pos

        request_data = {
            "requests": [
                {
                    "prompt": prompt,
                    "request_target": [request_target],
                },
            ]
        }
        print(f"request_data: {request_data}")

        # Pack request data using msgpack
        packed_data = msgpack.packb(request_data)

        # Prepare request type and request data
        request_type_data = np.array([["forward"]], dtype=np.object_)
        request_bytes_data = np.array([[packed_data]], dtype=np.object_)

        # Create input tensors
        inputs = [
            prepare_tensor(grpcclient, "request_type", request_type_data),
            prepare_tensor(grpcclient, "request_bytes", request_bytes_data),
        ]

        # Set outputs
        outputs = [
            grpcclient.InferRequestedOutput("forward_response"),
        ]

        # Create result callback function
        result_list = []

        def callback(result, error, user_data=None):
            if error:
                print(f"Error: {error}")
                return

            response_bytes = result.as_numpy("forward_response")[0]
            if response_bytes is None:
                return

            try:
                print(
                    f"Received response raw size: {len(response_bytes)} bytes")
                unpacked_response = msgpack.unpackb(response_bytes)
                print(
                    f"Unpacked response structure: {list(unpacked_response.keys())}")

                # Add response to result list and display it
                result_list.append(unpacked_response)
                show_response(unpacked_response)
            except (ValueError, KeyError, TypeError) as e:
                print(f"Error parsing response: {e}")
                print(f"Response may be malformed or incomplete")
            done_event.set()

        # Send request using stream mode
        client.start_stream(callback=callback)
        client.async_stream_infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs,
            request_id=str(request_id),
            sequence_id=1,
            sequence_start=True,
            sequence_end=True,
        )

        # Wait for response
        if not done_event.wait(timeout=10.0):
            print("Warning: Timed out waiting for response after 10 seconds")
        client.stop_stream()

        return result_list[0] if result_list else None


def parse_args():
    parser = argparse.ArgumentParser(
        description="KsanaLLM Triton forward request client")
    parser.add_argument(
        "--host", type=str, default="localhost", help="Server host address"
    )
    parser.add_argument("--port", type=int, default=8021,
                        help="Server gRPC port")
    parser.add_argument(
        "--prompt", type=str,
        default="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant",
        help="Input prompt text"
    )
    parser.add_argument("--model", type=str,
                        default="ksana_llm", help="Model name")
    parser.add_argument("--target", type=str, default="layernorm",
                        choices=["layernorm", "transformer", "logits"],
                        help="Target layer name")
    parser.add_argument("--token_id", type=int, default=[13], nargs="+",
                        help="Token IDs to retrieve")
    parser.add_argument("--token_reduce_mode", type=str, default="GATHER_ALL",
                        help="Token reduce mode")
    parser.add_argument("--slice_pos", type=str,
                        help="Slice position, format as [[start1,end1],[start2,end2]...]")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Generate request ID
    REQUEST_ID = str(np.random.randint(1, np.iinfo(np.int32).max))

    print(f"Input prompt:\n{args.prompt}")
    print(
        f"Target: {args.target}, token_reduce_mode: {args.token_reduce_mode}")

    # Process slice_pos parameter
    SLICE_POS = None
    if args.slice_pos:
        try:
            # Convert string to nested list, e.g. "[[0,1],[2,3]]"
            SLICE_POS = eval(args.slice_pos)
        except (SyntaxError, ValueError) as e:
            print(f"Invalid slice_pos format: {e}")
            print("Should be [[start1,end1],[start2,end2]...]")
            SLICE_POS = None

    start_time = time.time()

    URL = f"{args.host}:{args.port}"
    print(f"Sending request to {URL}")
    triton_forward(
        url=URL,
        model_name=args.model,
        prompt=args.prompt,
        request_id=REQUEST_ID,
        target_name=args.target,
        token_id=args.token_id,
        token_reduce_mode=args.token_reduce_mode,
        slice_pos=SLICE_POS
    )

    end_time = time.time()
    print(f"Request time: {end_time - start_time:.3f} seconds")
