# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
import os
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import orjson
import pytest


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


def create_mock_triton_utils():
    """Create a mock triton_python_backend_utils for testing"""
    mock_triton_utils = MagicMock()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    mock_triton_utils.get_model_dir.return_value = os.path.join(
        current_dir, "../../../../../../examples/"
    )
    return mock_triton_utils


@patch.dict('sys.modules', {
    'ksana_llm': MagicMock(),
    'ksana_llm.arg_utils': MagicMock(),
    'triton_python_backend_utils': create_mock_triton_utils()
})
def test_parse_input():
    # Import after patching
    from triton_python_backend_utils import pb_utils
    from model import parse_input

    # Mock Logger
    pb_utils.Logger = Mock()
    pb_utils.Logger.log_info = Mock()
    pb_utils.Logger.log_error = Mock()
    pb_utils.Logger.log_warn = Mock()

    # Mock triton_string_to_numpy
    pb_utils.triton_string_to_numpy = MagicMock(
        side_effect=mock_triton_string_to_numpy)

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

    streaming_tensor.as_numpy = Mock(
        return_value=np.full((1, 1), True, dtype=bool))

    request_output_len_tensor = Mock()
    request_output_len_tensor.name = Mock(
        return_value="request_output_len")
    request_output_len_tensor.as_numpy = Mock(
        return_value=np.array([[50]], dtype=np.uint32)
    )

    runtime_top_p = Mock()
    runtime_top_p.name = Mock(return_value="runtime_top_p")
    runtime_top_p.as_numpy = Mock(
        return_value=np.array([[0.12]], dtype=np.float32))

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
    # Move all patches inside the function to avoid decorator interference with pytest-asyncio
    with patch.dict('sys.modules', {
        'ksana_llm': MagicMock(),
        'ksana_llm.arg_utils': MagicMock(),
        'triton_python_backend_utils': create_mock_triton_utils()
    }), \
    patch("model.ksana_llm") as mock_ksana_llm, \
    patch("os.path.isfile") as mock_isfile, \
    patch("threading.Thread") as mock_thread, \
    patch("asyncio.get_event_loop") as mock_get_loop, \
    patch("asyncio.Event") as mock_event:
        
        # Setup basic mocks
        mock_isfile.return_value = True
        mock_loop = Mock()
        mock_thread_instance = Mock()
        mock_event_instance = Mock()
        
        mock_get_loop.return_value = mock_loop
        mock_thread.return_value = mock_thread_instance
        mock_event.return_value = mock_event_instance
        mock_thread_instance.start = Mock()
        
        # Import after patching
        from triton_python_backend_utils import pb_utils
        from model import TritonPythonModel

        # Setup pb_utils mocks
        pb_utils.Logger = Mock()
        pb_utils.Logger.log_info = Mock()
        pb_utils.Logger.log_error = Mock()
        pb_utils.Logger.log_warn = Mock()
        pb_utils.triton_string_to_numpy = MagicMock(side_effect=mock_triton_string_to_numpy)
        pb_utils.using_decoupled_model_transaction_policy = Mock(return_value=True)
        pb_utils.get_model_dir = Mock(return_value="/path/to/model")
        pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL = 1
        pb_utils.Tensor = Mock()
        pb_utils.InferenceResponse = Mock()
        pb_utils.TritonError = Exception

        # Setup ksana_llm mocks
        mock_engine = Mock()
        mock_engine.initialize = Mock()
        mock_engine.tokenizer.decode = Mock(return_value="Generated text.")
        mock_engine.generate = Mock(return_value=(True, iter([])))
        
        mock_engine_args = Mock()
        mock_engine_args.from_config_file = Mock(return_value=Mock())
        
        mock_ksana_llm.KsanaLLMEngine.from_engine_args = Mock(return_value=mock_engine)
        mock_ksana_llm.EngineArgs = mock_engine_args

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
            ],
            "output": [
                {"name": "text_output", "data_type": "TYPE_STRING", "dims": [-1]},
            ],
            "model_transaction_policy": {"decoupled": True},
            "max_batch_size": 1,
        }

        # Initialize model
        model = TritonPythonModel()
        args = {"model_config": orjson.dumps(model_config)}
        model.initialize(args)

        # Prepare input tensors
        text_input_tensor = Mock()
        text_input_tensor.name = Mock(return_value="text_input")
        text_input_tensor.as_numpy = Mock(return_value=np.array([b"Hello world"], dtype=object))

        streaming_tensor = Mock()
        streaming_tensor.name = Mock(return_value="streaming")
        streaming_tensor.as_numpy = Mock(return_value=np.array([True], dtype=bool))

        # Create mock request
        request = Mock()
        request.inputs = Mock(return_value=[text_input_tensor, streaming_tensor])
        response_sender = Mock()
        request.get_response_sender = Mock(return_value=response_sender)
        request.request_id = Mock(return_value="test_request_id")

        # Mock create_task to avoid asyncio issues
        def mock_create_task(coroutine_task):
            # Close the coroutine to avoid warnings
            coroutine_task.close()
            future_mock = Mock()
            future_mock.result = Mock()
            return future_mock
        
        model.create_task = Mock(side_effect=mock_create_task)

        # Execute test
        model.execute([request])

        # Verify
        assert model.create_task.called, "create_task was not called"
        print("create_task call arguments:", model.create_task.call_args_list)

        # Clean up
        model._shutdown_event = Mock()
        model._loop_thread = Mock()
        model._shutdown_event.set = Mock()
        model._loop_thread.join = Mock()
        model.finalize()


# Run the tests
if __name__ == "__main__":
    test_parse_input()
    # Note: test_model_execute() needs to be run with pytest due to async nature
    test_model_execute() 
