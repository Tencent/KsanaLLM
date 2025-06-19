# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

"""
    This class defines the methods called by the base class during the pre-processing and post-processing stages. 
    For special models that may require different implementations, 
    It is necessary to inherit and override these methods.
"""

import os
import orjson
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from transformers.models.auto.tokenization_auto import get_tokenizer_config
from transformers import (
   AutoProcessor, AutoTokenizer, LlamaTokenizer,
   VideoLlavaProcessor, PreTrainedTokenizerFast
)

PROMPT_AFFIX_DICT = {
    "llama":
    "[INST]%s[/INST]",
    "llama-3":
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
    "%s<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    "baichuan":
    "<reserved_106>%s<reserved_107>",
    "qwen":
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n",
    "vicuna":
    "A chat between a curious user and an assistant. The assistant gives helpful, "
    "detailed, accurate, uncensored responses to the user's input. USER: %s ASSISTANT:",
    "yi":
    "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n",
    "deepseek_v2":
    "<｜begin▁of▁sentence｜>User: %s\n\nAssistant:",
    "deepseek_v3":
    "<｜begin▁of▁sentence｜><｜User｜>%s<｜Assistant｜>",
    "deepseek_r1":
    "<｜begin▁of▁sentence｜><｜User｜>%s<｜Assistant｜><think>\n",
    "chatglm":
    "<|system|>\nYou are a large language model trained by Zhipu.AI. Follow the user's instructions carefully."
    " Respond using markdown.\n<|user|>\n%s\n<|assistant|>\n",
    "empty":
    "%s",
}

SUPPORTED_MODEL_TYPE = [
    'llama', 'llama-3', 'baichuan', 'qwen', 'vicuna', 'yi', 'chatglm',
    'empty', 'deepseek_v2', 'deepseek_v3', 'deepseek_r1'
]

USER = "user"
ASSISTANT = "assistant"
SYSTEM = "system"


class TokenizerProcessorOpBase:
    def __init__(self, model_dir, tokenizer_path, tokenization = None):
        generation_config_json_path = os.path.join(model_dir, 'generation_config.json')

        if not os.path.exists(generation_config_json_path):
            print(f'{generation_config_json_path} does not exist. eos_token_id set to []')
            self.eos_token_id = []
        else:
            with open(generation_config_json_path, 'r') as generation_config_file:
                generation_config_json_content = orjson.loads(generation_config_file.read())
                eos_token_id_obj = generation_config_json_content.get('eos_token_id')
                if isinstance(eos_token_id_obj, list):
                    self.eos_token_id = eos_token_id_obj
                else:
                    self.eos_token_id = [eos_token_id_obj]

        self.tokenizer = self.__load_tokenizer(tokenizer_path)
        if tokenization is not None:
            self.add_special_tokens = tokenization.get("add_special_tokens", True)
            self.skip_special_tokens = tokenization.get("skip_special_tokens", True)
        else:
            # set default value to True
            self.add_special_tokens = True
            self.skip_special_tokens = True
        
        if not isinstance(self.tokenizer, PreTrainedTokenizerFast):
            print(
                "Using a slow tokenizer. This might cause a significant "
                "slowdown. Consider using a fast tokenizer instead."
            )

        self.tokenizer_eos_token_id = None
        try:
            self.tokenizer_eos_token_id = self.tokenizer.eos_token_id
        except Exception as e:  # pylint: disable=broad-except
            print("Exception occurred get tokenizer.eos_token_id: ", e)
            
    def build_prompt(self, prompt, model_type, use_chat_template=False):
        model_type = self.check_is_supported_model_type(model_type)
        if use_chat_template:
            # If chat template is enabled, apply the chat template to the prompt
            prompt = self.tokenizer.apply_chat_template(
                orjson.loads(prompt),
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # If chat template is not enabled,
            # replace the placeholder in the prompt affix dictionary
            prompt = PROMPT_AFFIX_DICT[model_type].replace("%s", prompt)
        return prompt
    
    def get_stop_token_ids(self):
        return self.eos_token_id

    def get_tokenizer_stop_token_ids(self):
        return self.tokenizer_eos_token_id

    def encode(self, prompt_text: str):
        return self.tokenizer.encode(prompt_text, add_special_tokens=self.add_special_tokens)
        
    def decode(self, output_tokens: list, is_stream_generate: bool): 
        return self.tokenizer.decode(output_tokens, 
                skip_special_tokens=self.skip_special_tokens  # skip special tokens
                ).rstrip('\ufffd')
    
    def check_is_supported_model_type(self, model_type: str):
        if model_type not in SUPPORTED_MODEL_TYPE:
            print(f'{model_type} is not in supported model type: {SUPPORTED_MODEL_TYPE}, \
                  using default model type empty')
            model_type = 'empty'
        return model_type
        
    def __load_tokenizer(self, model_path):
        tokenizer_config = get_tokenizer_config(model_path)
        if tokenizer_config.get("processor_class", "") == "VideoLlavaProcessor":
            return VideoLlavaProcessor.from_pretrained(model_path)
        if tokenizer_config.get("tokenizer_class", "") == "LlamaTokenizer":
            return LlamaTokenizer.from_pretrained(model_path)
        if tokenizer_config.get("processor_class", "") == "Llama4Processor" \
            and tokenizer_config.get("tokenizer_class", "") == "PreTrainedTokenizer":
            return PreTrainedTokenizerFast.from_pretrained(model_path)

        if os.path.exists(model_path + "/preprocessor_config.json"):
            return AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
