# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
import argparse
import os
from dataclasses import dataclass
from typing import Optional, Dict
import yaml
from transformers import AutoConfig


@dataclass
class EngineArgs:
    config_file: str
    model_dir: str
    tokenizer_path: str
    model_type: str
    endpoint: str
    host: str
    port: int
    access_log: bool
    max_token_len: int = 2048
    plugin_model_enable_trt: bool = True
    plugin_thread_pool_size: int = 1
    ssl_keyfile: Optional[str] = None
    ssl_certfile: Optional[str] = None
    root_path: Optional[str] = None
    tokenization: Optional[Dict[str, Dict[str, bool]]] = None
    multi_modal_vit_model_type: Optional[str] = None
    # OpenAI Adapter related parameters
    tool_call_parser: Optional[str] = None
    reasoning_parser: Optional[str] = None
    enable_auto_tool_choice: bool = False
    tool_parser_plugin: Optional[str] = None
    @classmethod
    def from_config_file(cls, config_file: str) -> "EngineArgs":
        if not os.path.exists(config_file):
            raise RuntimeError(f"The config file {config_file} does not exist.")

        if not config_file.lower().endswith(".yaml"):
            raise RuntimeError(f"The config file {config_file} must be in YAML format.")

        with open(config_file, "r") as yaml_file:
            yaml_config = yaml.safe_load(yaml_file)

        # Extract configurations from the YAML file
        model_dir = os.path.abspath(
            yaml_config["model_spec"]["base_model"]["model_dir"]
        )
        tokenizer_path = (
            model_dir  # Default tokenizer_path; can be overridden by CLI args
        )

        tokenization = yaml_config["setting"].get("tokenization", None)
        if tokenization is not None and tokenization.get("tokenizer_path", None) is not None:
            tokenizer_path = os.path.abspath(tokenization["tokenizer_path"])

        plugin_model_enable_trt = (
            yaml_config["model_spec"]
            .get("plugin_model", {})
            .get("enable_tensorrt", True)
        )

        plugin_thread_pool_size = (
            yaml_config["model_spec"]
            .get("plugin_model", {})
            .get("thread_pool_size", 1)
        )

        endpoint = yaml_config["setting"].get("endpoint_type", "python").lower()

        max_token_len = yaml_config["setting"]["batch_scheduler"].get(
            "max_token_len", 2048
        )

        # Temporary workaround for DeepSeek-V3.2 compatibility
        # Use the same config class as DeepSeek-V3
        # Can be removed once hf-transformers adds support
        try:
            model_config = AutoConfig.from_pretrained(
                model_dir, trust_remote_code=True
            ).to_dict()
        except ValueError as e:
            if not "deepseek_v32" in str(e):
                raise e
            from .hf_transformers_model_config import DeepseekV3Config
            model_config = DeepseekV3Config.from_pretrained(
                model_dir, trust_remote_code=True
            ).to_dict()
        # Parse the model config to determine the model type
        model_type = model_config["model_type"]
        if model_type == "qwen" and "visual" in model_config:
            model_type = "qwen_vl"

        # Adjust model type for internlmxcomposer2 and internvl_chat
        def determine_model_and_vit_model_type(model_type: str, model_config: dict) -> tuple:
            """根据模型配置确定模型类型和视觉模型类型"""
            vit_model_type = None
            if model_type == "internlm2" \
                and "InternLMXComposer2ForCausalLM" in model_config["architectures"]:
                model_type = "internlmxcomposer2"
            elif model_type == "internvl_chat":
                if model_config['template'] == "internvl2_5":
                    print(f'current model type is: InternVL2_5')
                    vit_model_type = "InternVL2_5"
                elif model_config['template'] == "internlm2-chat":
                    print(f'current model type is: InternVL2')
                    vit_model_type = "InternVL2"
                else:
                    raise ValueError(f'current model type is not supported: {model_config["template"]}')
            return model_type, vit_model_type

        model_type, vit_model_type = determine_model_and_vit_model_type(model_type, model_config)
        multi_modal_vit_model_type = (
            vit_model_type
        )

        # Initialize EngineArgs with default values from the config file
        return cls(
            config_file=config_file,
            model_dir=model_dir,
            tokenizer_path=tokenizer_path,
            model_type=model_type,
            endpoint=endpoint,
            host="localhost",  # Default host; can be overridden by CLI args
            port=8080,  # Default port; can be overridden by CLI args
            access_log=True,
            max_token_len=max_token_len,
            plugin_model_enable_trt=plugin_model_enable_trt,
            plugin_thread_pool_size=plugin_thread_pool_size,
            multi_modal_vit_model_type=multi_modal_vit_model_type
        )

    @classmethod
    def from_command_line(cls) -> "EngineArgs":
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--config_file",
            type=str,
            default="examples/ksana_llm.yaml",
            help="Serving config file.",
        )
        parser.add_argument(
            "--tokenizer_path", type=str, default=None, help="Tokenizer directory."
        )
        parser.add_argument(
            "--host", type=str, default=None, help="Server host address."
        )
        parser.add_argument("--port", type=int, default=None, help="Server port.")
        parser.add_argument(
            "--endpoint",
            type=str,
            default=None,
            help="Server endpoint type (e.g., 'python' or 'trpc').",
        )
        parser.add_argument(
            "--access_log",
            action="store_true",
            help="Enable the endpoint access log.",
        )
        parser.add_argument("--ssl_keyfile", type=str, default=None)
        parser.add_argument("--ssl_certfile", type=str, default=None)
        parser.add_argument(
            "--root_path",
            type=str,
            default=None,
            help="FastAPI root_path when app is behind a path-based routing proxy.",
        )
        # OpenAI Adapter related parameters
        parser.add_argument(
            "--tool-call-parser",
            type=str,
            default=None,
            help="Tool call parser to use for parsing tool calls in chat completion. "
                 "Choices: ['mistral', 'internlm2', 'llama3_json', 'llama4_pythonic', "
                 "'pythonic', 'deepseekv3']",
        )
        parser.add_argument(
            "--reasoning-parser",
            type=str,
            default=None,
            help="Reasoning parser to use for parsing reasoning content. "
                 "Choices: ['deepseek_r1', 'granite', 'qwen3']",
        )
        parser.add_argument(
            "--enable-auto-tool-choice",
            action="store_true",
            default=False,
            help="Enable auto tool choice for tool calls.",
        )
        parser.add_argument(
            "--tool-parser-plugin",
            type=str,
            default=None,
            help="Path to a custom tool parser plugin file.",
        )
        parser.add_argument(
            "--chat-template",
            type=str,
            default=None,
            help="Path to a custom chat template file.",
        )
        parser.add_argument(
            "--chat-template-content-format",
            type=str,
            default=None,
            help="Content format for the chat template. "
        )
        # Default values for CLI arguments
        parser.set_defaults(access_log=True)

        args, unknown = parser.parse_known_args()
        print(f"Arguments: {args}")
        if unknown:
            print(f"Unknown arguments: {unknown}")

        # Initialize EngineArgs from the config file
        engine_args = cls.from_config_file(args.config_file)

        # Override attributes if provided via CLI arguments
        if args.tokenizer_path is not None:
            engine_args.tokenizer_path = args.tokenizer_path
        if args.host is not None:
            engine_args.host = args.host
        if args.port is not None:
            engine_args.port = args.port
        if args.endpoint is not None:
            engine_args.endpoint = args.endpoint.lower()
        if args.access_log is not None:
            engine_args.access_log = args.access_log
        if args.ssl_keyfile is not None:
            engine_args.ssl_keyfile = args.ssl_keyfile
        if args.ssl_certfile is not None:
            engine_args.ssl_certfile = args.ssl_certfile
        if args.root_path is not None:
            engine_args.root_path = args.root_path

        # OpenAI Adapter related parameters
        if args.tool_call_parser is not None:
            engine_args.tool_call_parser = args.tool_call_parser
        if args.reasoning_parser is not None:
            engine_args.reasoning_parser = args.reasoning_parser
        if args.enable_auto_tool_choice is not None:
            engine_args.enable_auto_tool_choice = args.enable_auto_tool_choice
        if args.tool_parser_plugin is not None:
            engine_args.tool_parser_plugin = args.tool_parser_plugin
        if args.chat_template is not None:
            engine_args.chat_template = args.chat_template
        if args.chat_template_content_format is not None:
            engine_args.chat_template_content_format = args.chat_template_content_format

        return engine_args
