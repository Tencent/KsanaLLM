import json
import math
from typing import List, Optional, Union
from multiprocessing import Pool
import logging
from transformers import AutoTokenizer
from tqdm import tqdm

# When chatting with models, the system template prompt will be inserted into each user prompt
SYSTEM_TOKEN_LEN = 100

# Prompt templates, ref: https://github.com/THUDM/LongBench/tree/2e00731/prompts
PROMPT_TEMPLATES = {
    # 0shot prompt template with long context (ref: 0shot.txt)
    "with_context": (
        "Please read the following text and answer the question below.\n"
        "\n"
        "<text>\n"
        "$DOC$\n"
        "</text>\n"
        "\n"
        "What is the correct answer to this question: $Q$\n"
        "Choices:\n"
        "(A) $C_A$\n"
        "(B) $C_B$\n"
        "(C) $C_C$\n"
        "(D) $C_D$\n"
        "\n"
        "Format your response as follows: \"The correct answer is (insert answer here)\".\n"
    ),
    # 0shot prompt template without context (ref: 0shot_no_context.txt)
    "no_context": (
        "What is the correct answer to this question: $Q$\n"
        "Choices:\n"
        "(A) $C_A$\n"
        "(B) $C_B$\n"
        "(C) $C_C$\n"
        "(D) $C_D$\n"
        "\n"
        "What is the single, most likely answer choice? Format your response as follows: "
        "\"The correct answer is (insert answer here)\".\n"
    )
}


def _process_batch_with_init(args):
    """
    Process a batch of prompts in each worker process.

    Args:
        args: A tuple containing:
            - tokenizer_path: Path to load the tokenizer.
            - batch_prompts: List of prompts to process in this batch.
    Returns:
        A tuple containing:
            - List of processed prompts.
            - Error message if any occurred during processing, otherwise None.
    """
    tokenizer_path, batch_prompts = args

    try:
        local_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,
                                                        trust_remote_code=True,
                                                        use_fast=True)
        max_len = local_tokenizer.model_max_length - SYSTEM_TOKEN_LEN
    except (OSError, ValueError) as e:
        return [], f"Failed to load tokenizer from {tokenizer_path}: {str(e)}"

    logger = logging.getLogger("transformers")
    original_report_level = logger.level
    try:
        logger.setLevel(
            logging.ERROR)  # Suppress tokenizer warnings for cleaner output
        batch_input_ids = local_tokenizer(batch_prompts,
                                          truncation=False,
                                          padding=False,
                                          return_attention_mask=False,
                                          add_special_tokens=True)
        processed_batch = []
        need_decode_ids = []

        for i, input_ids in enumerate(batch_input_ids['input_ids']):
            if len(input_ids) > max_len:
                # truncate, ref: https://github.com/THUDM/LongBench/blob/2e00731/pred.py#L24-L35
                half_len = max_len // 2
                need_decode_ids.append(input_ids[:half_len] +
                                       input_ids[-half_len:])
            else:
                processed_batch.append(batch_prompts[i])

        # Decode the input_ids that need decoding
        if need_decode_ids:
            decoded_prompts = local_tokenizer.batch_decode(
                need_decode_ids, skip_special_tokens=True)
            processed_batch.extend(decoded_prompts)

        return processed_batch, None
    except (ValueError, OSError) as e:
        return [], f"Failed to process batch: {str(e)}"
    finally:
        logger.setLevel(original_report_level)


class LongBenchV2Dataset:

    def __init__(self,
                 file_path: Optional[str] = None,
                 with_context: bool = True,
                 tokenizer_path: Union[None, str] = None):
        """
        Initialize the LongBench V2 dataset object.
        
        Args:
            file_path: Path to a local JSON file containing the dataset.
                       If None, the dataset will be loaded from HuggingFace.
            with_context: Whether to use the template with context or without context.
            tokenizer_path: Optional path to the tokenizer to use for processing prompts.
        """
        self.template = PROMPT_TEMPLATES[
            "with_context" if with_context else "no_context"]
        self.max_input_len = 120000
        self.set_tokenizer(tokenizer_path)

        self.logger = logging.getLogger("transformers")
        self.report_level = self.logger.level

        if file_path:
            if not file_path.endswith('.json'):
                raise ValueError(
                    "LongBench V2 dataset file path must end with .json")

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                raise ValueError(
                    f"Failed to load dataset from {file_path}: {e}") from e
        else:
            try:
                from datasets import load_dataset
                self.data = load_dataset('THUDM/LongBench-v2', split='train')
            except Exception as e:
                raise ValueError(
                    f"Failed to load dataset from HuggingFace: {e}") from e

        self.prompts = []

        for item in self.data:
            self.prompts.append(self._construct_prompt(item))

    def set_tokenizer(self, tokenizer_path: Union[None, str]):
        self.tokenizer_path = tokenizer_path
        if self.tokenizer_path is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, trust_remote_code=True, use_fast=True)
            self.max_input_len = self.tokenizer.model_max_length - SYSTEM_TOKEN_LEN

    def _construct_prompt(self, question):
        """
        Construct a prompt based on the template and question data.
        """
        prompt = self.template.replace('$DOC$', question['context'].strip())
        prompt = prompt.replace('$Q$', question['question'].strip())
        prompt = prompt.replace('$C_A$', question['choice_A'].strip())
        prompt = prompt.replace('$C_B$', question['choice_B'].strip())
        prompt = prompt.replace('$C_C$', question['choice_C'].strip())
        prompt = prompt.replace('$C_D$', question['choice_D'].strip())
        return prompt

    def get_all_prompts(self,
                        tokenizer_path: Union[None, str] = None,
                        num_workers=8) -> List[str]:
        """
        Get all processed prompts.

        Args:
            tokenizer_path: Optional path to the tokenizer to check for overly long prompts.
                           If None, will use the tokenizer_path set in the class.
            num_workers: Number of worker processes to use for parallel processing.

        Returns:
            List[str]: A list of processed prompts.
        """

        if tokenizer_path is None:
            tokenizer_path = self.tokenizer_path
            if tokenizer_path is None:
                raise ValueError(
                    "tokenizer_path should not be None. Please set it before calling get_all_prompts."
                )

        prompt_num = len(self.prompts)
        batch_size = max(1, prompt_num // num_workers)
        num_batches = math.ceil(prompt_num / batch_size)
        batches = []
        for start_idx in range(0, prompt_num, batch_size):
            # for each batch (process), prepare: tokenizer path, prompts batch
            batches.append((tokenizer_path,
                            self.prompts[start_idx:start_idx + batch_size]))

        progress_bar = tqdm(total=num_batches,
                            desc="Processing Prompts",
                            unit="batch")
        all_processed_prompts = []

        with Pool(processes=num_workers) as pool:
            for batch_results, error in pool.imap(_process_batch_with_init,
                                                  batches):
                if error:
                    print(error)
                else:
                    # extend the results to the final list
                    all_processed_prompts.extend(batch_results)
                progress_bar.update(1)

        progress_bar.close()
        return all_processed_prompts

    def get_prompt(self,
                   index: int,
                   tokenizer_path: Union[None, str] = None) -> str:
        """
        Get a single prompt by index.
        
        Args:
            index: Index of the prompt to retrieve.
            tokenizer_path: Optional path to the tokenizer to check for overly long prompts.
                           If None, will use the tokenizer_path set in the class.
        Returns:
            str: The processed prompt at the specified index.
        """
        if index < 0 or index >= len(self.prompts):
            raise IndexError("Index out of range.")

        if tokenizer_path is None:
            tokenizer_path = self.tokenizer_path
            if self.tokenizer_path is None:
                raise ValueError(
                    "tokenizer_path should not be None. Please set it before calling get_prompt."
                )
        else:
            if tokenizer_path != self.tokenizer_path:
                self.set_tokenizer(tokenizer_path)

        try:
            self.logger.setLevel(logging.ERROR)
            tokenizer_result = self.tokenizer(self.prompts[index],
                                              truncation=False,
                                              padding=False,
                                              return_attention_mask=False,
                                              add_special_tokens=True)
            if len(tokenizer_result['input_ids']) > self.max_input_len:
                half_len = self.max_input_len // 2
                truncated_ids = tokenizer_result[
                    'input_ids'][:half_len] + tokenizer_result['input_ids'][-half_len:]
                return self.tokenizer.decode(truncated_ids,
                                             skip_special_tokens=True)
            else:
                return self.prompts[index]
        except Exception as e:
            raise ValueError(f"Failed to process prompt at index {index}: {e}") from e
        finally:
            self.logger.setLevel(self.report_level)

    def __len__(self) -> int:
        """
        Get the number of prompts in the dataset.
        """
        return len(self.data)

    def __getitem__(self, index: int) -> str:
        return self.get_prompt(index)
