# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
import sys
import os
from unittest.mock import Mock, MagicMock, patch
import pytest
import numpy as np
import orjson

# Mock triton_python_backend_utils and ksana_llm\

mock_triton_utils = MagicMock()
current_dir = os.path.dirname(os.path.abspath(__file__))
mock_triton_utils.get_model_dir.return_value = os.path.join(
    current_dir, "../../../../../../examples/"
)

# 断点，继续研究mock
# Mock pb_utils in triton_python_backend_utils
sys.modules["triton_python_backend_utils"] = mock_triton_utils
sys.modules["ksana_llm"] = MagicMock()
sys.modules["ksana_llm.arg_utils"] = MagicMock()


# Now import the model code
from model import TritonPythonModel, parse_input


def mock_triton_string_to_numpy(data_type_str):
    mapping = {
        "TYPE_BOOL": np.bool_,
        "TYPE_FP32": np.float32,
        "TYPE_INT32": np.int32,
        "TYPE_UINT32": np.uint32,
        "TYPE_UINT64": np.uint64,
        "TYPE_STRING": np.object_,
    }
    return mapping.get(data_type_str, np.object_)


def test_parse_input():
    # Mock the pb_utils module
    from triton_python_backend_utils import pb_utils

    # Mock Logger
    pb_utils.Logger = Mock()
    pb_utils.Logger.log_info = Mock()
    pb_utils.Logger.log_error = Mock()
    pb_utils.Logger.log_warn = Mock()

    # Mock triton_string_to_numpy
    pb_utils.triton_string_to_numpy = MagicMock(side_effect=mock_triton_string_to_numpy)

    # Define input configurations as per config.pbtxt
    input_config_list = [
        {
            "name": "text_input",
            "data_type": "TYPE_STRING",
            "dims": [1],
            "optional": True,
        },
        {
            "name": "input_ids",
            "data_type": "TYPE_UINT32",
            "dims": [-1],
            "optional": True,
        },
        {"name": "streaming", "data_type": "TYPE_BOOL", "dims": [1]},
        {
            "name": "runtime_top_p",
            "data_type": "TYPE_FP32",
            "dims": [1],
            "optional": True,
        },
        {
            "name": "request_output_len",
            "data_type": "TYPE_UINT32",
            "dims": [1],
            "optional": True,
        },
        # Add other inputs as needed...
    ]

    # Build input_dtypes dictionary
    input_dtypes = {
        input_cfg["name"]: (
            pb_utils.triton_string_to_numpy(input_cfg["data_type"]),
            input_cfg["dims"],
        )
        for input_cfg in input_config_list
    }

    # Prepare mock input tensors
    text_input_tensor = Mock()
    text_input_tensor.name = Mock(return_value="text_input")
    text_input_tensor.as_numpy = Mock(
        return_value=np.array([[b"Hello world"]], dtype=object)
    )

    streaming_tensor = Mock()
    streaming_tensor.name = Mock(return_value="streaming")

    streaming_tensor.as_numpy = Mock(return_value=np.full((1, 1), True, dtype=bool))

    request_output_len_tensor = Mock()
    request_output_len_tensor.name = Mock(return_value="request_output_len")
    request_output_len_tensor.as_numpy = Mock(
        return_value=np.array([[50]], dtype=np.uint32)
    )

    runtime_top_p = Mock()
    runtime_top_p.name = Mock(return_value="runtime_top_p")
    runtime_top_p.as_numpy = Mock(return_value=np.array([[0.12]], dtype=np.float32))

    # Create a mock InferenceRequest
    request = Mock()
    request.inputs = Mock(
        return_value=[
            text_input_tensor,
            streaming_tensor,
            request_output_len_tensor,
            runtime_top_p,
        ]
    )

    # Call parse_input function
    request_dict = parse_input(pb_utils.Logger, request, input_dtypes)
    print(f"request_dict is: {request_dict}")
    # Verify that the inputs are parsed correctly
    assert request_dict["prompt"] == "Hello world"
    assert request_dict["streaming"] is True

    assert (
        request_dict["max_new_tokens"] == 50
    )  # 'request_output_len' maps to 'max_new_tokens'
    print("Parsed request dictionary:", request_dict)


@pytest.mark.asyncio
async def test_model_execute():
    # Mock the pb_utils module
    from triton_python_backend_utils import pb_utils

    # Mock Logger
    pb_utils.Logger = Mock()
    pb_utils.Logger.log_info = Mock()
    pb_utils.Logger.log_error = Mock()
    pb_utils.Logger.log_warn = Mock()

    # Mock functions and constants
    pb_utils.triton_string_to_numpy = MagicMock(side_effect=mock_triton_string_to_numpy)
    pb_utils.using_decoupled_model_transaction_policy = Mock(return_value=True)
    pb_utils.get_model_dir = Mock(return_value="/path/to/model")
    pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL = 1
    pb_utils.Tensor = Mock()
    pb_utils.InferenceResponse = Mock()
    pb_utils.TritonError = Exception

    # Prepare model_config
    model_config = {
        "input": [
            {
                "name": "text_input",
                "data_type": "TYPE_STRING",
                "dims": [1],
                "optional": True,
            },
            {"name": "streaming", "data_type": "TYPE_BOOL", "dims": [1]},
            # Add other inputs as needed...
        ],
        "output": [
            {"name": "text_output", "data_type": "TYPE_STRING", "dims": [-1]},
            # Add other outputs as needed...
        ],
        "model_transaction_policy": {"decoupled": True},
        "max_batch_size": 1,
    }

    # Instantiate the model
    model = TritonPythonModel()
    args = {
        "model_config": orjson.dumps(model_config),
    }

    # Mock ksana_llm and its components
    with patch("model.ksana_llm") as mock_ksana_llm:
        # Mock os.path.isfile to return True for ksana_llm.yaml
        with patch("os.path.isfile") as mock_isfile:

            def isfile_side_effect(path):
                if path.endswith("_KSANA_CONFIG_FILENAME"):
                    return True
                else:
                    return False

            mock_isfile.side_effect = isfile_side_effect

        # Mock the KsanaLLMEngine
        mock_engine = Mock()
        mock_engine.initialize = Mock()
        mock_engine.tokenizer.decode = Mock(return_value="Generated text.")
        mock_engine.generate = Mock(return_value=(True, iter([])))

        # Mock ksana_llm.EngineArgs.from_config_file
        mock_engine_args = Mock()
        mock_engine_args.from_config_file = Mock(return_value=Mock())

        # Setup the ksana_llm mocks
        mock_ksana_llm.KsanaLLMEngine.from_engine_args = Mock(return_value=mock_engine)
        mock_ksana_llm.EngineArgs = mock_engine_args

        # Initialize the model
        model.initialize(args)

        # Prepare input tensors
        text_input_tensor = Mock()
        text_input_tensor.name = Mock(return_value="text_input")
        text_input_tensor.as_numpy = Mock(
            return_value=np.array([b"Hello world"], dtype=object)
        )

        streaming_tensor = Mock()
        streaming_tensor.name = Mock(return_value="streaming")
        streaming_tensor.as_numpy = Mock(return_value=np.array([True], dtype=bool))

        # Create a mock InferenceRequest
        request = Mock()
        request.inputs = Mock(return_value=[text_input_tensor, streaming_tensor])
        response_sender = Mock()
        request.get_response_sender = Mock(return_value=response_sender)
        request.request_id = Mock(return_value="test_request_id")

        # Mock the generate_per_req coroutine
        async def mock_generate_per_req(response_sender, request_dict, req_id):
            # Simulate sending a response
            response_sender.send(Mock())

        # Patch the generate_per_req method
        model.generate_per_req = mock_generate_per_req

        # Call execute with the mock request
        model.execute([request])

        # Allow scheduled coroutines to run
        await asyncio.sleep(0.1)  # Adjust the sleep time as needed

        # Verify that response_sender.send was called
        assert (
            response_sender.send.called
        ), "The response sender's send method was not called."

        # Print out the call arguments for debugging
        print(
            "Response sender send call arguments:", response_sender.send.call_args_list
        )


# Run the tests
if __name__ == "__main__":
    test_parse_input()
    test_model_execute()
