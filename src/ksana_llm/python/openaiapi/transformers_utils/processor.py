# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Ref:
# https://github.com/vllm-project/vllm/blob/v0.9.1/vllm/transformers_utils/processor.py
from functools import lru_cache
from typing import Any, Union

from transformers.processing_utils import ProcessorMixin
from typing_extensions import TypeVar


_P = TypeVar("_P", bound=ProcessorMixin, default=ProcessorMixin)


class HashableDict(dict):
    """
    A dictionary that can be hashed by lru_cache.
    """

    # NOTE: pythonic dict is not hashable,
    # we override on it directly for simplicity
    def __hash__(self) -> int:  # type: ignore[override]
        return hash(frozenset(self.items()))


class HashableList(list):
    """
    A list that can be hashed by lru_cache.
    """

    def __hash__(self) -> int:  # type: ignore[override]
        return hash(tuple(self))


def get_processor(
    processor_name: str,
    *args: Any,
    trust_remote_code: bool = False,
    processor_cls: Union[type[_P], tuple[type[_P], ...]] = ProcessorMixin,
    **kwargs: Any,
) -> _P:
    """Load a processor for the given model name via HuggingFace."""
    # don't put this import at the top level
    # it will call torch.cuda.device_count()
    from transformers import AutoProcessor

    processor_factory = (AutoProcessor if processor_cls == ProcessorMixin or
                         isinstance(processor_cls, tuple) else processor_cls)

    try:
        processor = processor_factory.from_pretrained(
            processor_name,
            *args,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
    except ValueError as e:
        # If the error pertains to the processor class not existing or not
        # currently being imported, suggest using the --trust-remote-code flag.
        # Unlike AutoTokenizer, AutoProcessor does not separate such errors
        if not trust_remote_code:
            err_msg = (
                "Failed to load the processor. If the processor is "
                "a custom processor not yet available in the HuggingFace "
                "transformers library, consider setting "
                "`trust_remote_code=True` in LLM or using the "
                "`--trust-remote-code` flag in the CLI.")
            raise RuntimeError(err_msg) from e
        else:
            raise e

    if not isinstance(processor, processor_cls):
        raise TypeError("Invalid type of HuggingFace processor. "
                        f"Expected type: {processor_cls}, but "
                        f"found type: {type(processor)}")

    return processor


cached_get_processor = lru_cache(get_processor)
