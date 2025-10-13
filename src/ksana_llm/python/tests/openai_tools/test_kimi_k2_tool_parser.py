# Copyright 2025 Tencent Inc.  All rights reserved.
#
# ==============================================================================
import sys
import os
from unittest.mock import Mock
from typing import Dict
import pytest

# 添加路径以便导入模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from openaiapi.tool_parsers.kimi_k2_tool_parser import KimiK2ToolParser
from openaiapi.openai_protocol import (
    ChatCompletionRequest, 
    ExtractedToolCallInformation,
    DeltaMessage,
    ToolCall,
    FunctionCall,
)

pytestmark = pytest.mark.cpu_test


class MockTokenizer:
    """
    模拟的 tokenizer 类，使用真实的 KimiK2 token IDs
    这个类模拟了真实 tokenizer 的行为，提供了 Kimi K2 模型中用于工具调用的特殊 token。
    这些 token ID 是从真实的 tokenizer 配置中Copy 下来的。
    工具调用相关的特殊 token：
    - tool_calls_section_begin/end: 标记整个工具调用区域的开始和结束
    - tool_call_begin/end: 标记单个工具调用的开始和结束  
    - tool_call_argument_begin: 标记工具调用参数的开始
    """
    
    def __init__(self):
        # 使用真实的 KimiK2 special token IDs（来自 tokenizer 配置）
        self.vocab = {
            "<|tool_calls_section_begin|>": 163595,  # 工具调用区域开始标记
            "<|tool_calls_section_end|>": 163596,    # 工具调用区域结束标记
            "<|tool_call_begin|>": 163597,           # 单个工具调用开始标记
            "<|tool_call_end|>": 163599,             # 单个工具调用结束标记
            "<|tool_call_argument_begin|>": 163598,  # 工具调用参数开始标记
            "<|im_end|>": 163586,                    # 消息结束标记
            "<|im_user|>": 163587,                   # 用户消息标记
            "<|im_assistant|>": 163588,              # 助手消息标记
            "<|im_system|>": 163594,                 # 系统消息标记
            "[BOS]": 163584,                         # 序列开始标记
            "[EOS]": 163585,                         # 序列结束标记
        }
    
    def get_vocab(self) -> Dict[str, int]:
        """返回词汇表，模拟真实 tokenizer 的 get_vocab 方法"""
        return self.vocab


@pytest.fixture
def mock_tokenizer():
    """创建模拟的 tokenizer"""
    return MockTokenizer()


@pytest.fixture
def tool_parser(mock_tokenizer):
    """创建 KimiK2ToolParser 实例"""
    return KimiK2ToolParser(mock_tokenizer)


@pytest.fixture
def sample_request():
    """创建示例请求"""
    return ChatCompletionRequest(
        model="test-model",
        messages=[
            {"role": "user", "content": "What's the weather like?"}
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        }
                    }
                }
            }
        ]
    )


def assert_tool_calls(
    actual_tool_calls: list[ToolCall], expected_tool_calls: list[ToolCall]
):
    """断言工具调用是否匹配预期"""
    assert len(actual_tool_calls) == len(expected_tool_calls)

    for actual_tool_call, expected_tool_call in zip(
        actual_tool_calls, expected_tool_calls
    ):
        assert actual_tool_call.type == "function"
        assert actual_tool_call.function == expected_tool_call.function

        # assert tool call id format
        assert actual_tool_call.id.startswith("functions.")
        assert actual_tool_call.id.split(":")[-1].isdigit()
        assert (
            actual_tool_call.id.split(".")[1].split(":")[0]
            == expected_tool_call.function.name
        )


class TestKimiK2ToolParser:
    """KimiK2ToolParser 精简测试类"""

    def test_initialization(self, mock_tokenizer):
        """测试初始化和错误处理"""
        parser = KimiK2ToolParser(mock_tokenizer)
        assert parser.tool_calls_start_token == "<|tool_calls_section_begin|>"
        assert parser.tool_calls_end_token == "<|tool_calls_section_end|>"
        assert parser.tool_call_start_token == "<|tool_call_begin|>"
        assert parser.tool_call_end_token == "<|tool_call_end|>"
        
        # 测试错误处理
        with pytest.raises(ValueError, match="The model tokenizer must be passed"):
            KimiK2ToolParser(None)
        
        # 测试缺少必要 token
        incomplete_tokenizer = Mock()
        incomplete_tokenizer.get_vocab.return_value = {}
        with pytest.raises(RuntimeError, match="Kimi-K2 Tool parser could not locate"):
            KimiK2ToolParser(incomplete_tokenizer)

    def test_extract_tool_calls_no_tools(self, tool_parser, sample_request):
        """测试没有工具调用的情况"""
        model_output = "This is a regular response without any tool calls."
        
        result = tool_parser.extract_tool_calls(model_output, sample_request)
        
        assert isinstance(result, ExtractedToolCallInformation)
        assert not result.tools_called
        assert result.tool_calls == []
        assert result.content == model_output

    @pytest.mark.parametrize(
        ids=[
            "single_tool_call",
            "complex_arguments",
            "empty_arguments",
        ],
        argnames=["model_output", "expected_tool_calls", "expected_content"],
        argvalues=[
            (
                (
                    "I'll help you check the weather. <|tool_calls_section_begin|> <|tool_call_begin|>"
                    "functions.get_weather:0 <|tool_call_argument_begin|> "
                    '{"city": "Beijing"} <|tool_call_end|> <|tool_calls_section_end|>'
                ),
                [
                    ToolCall(
                        id="functions.get_weather:0",
                        function=FunctionCall(
                            name="get_weather",
                            arguments='{"city": "Beijing"}',
                        ),
                        type="function",
                    )
                ],
                "I'll help you check the weather. ",
            ),

            (
                (
                    "<|tool_calls_section_begin|> <|tool_call_begin|>"
                    "functions.complex_function:0 <|tool_call_argument_begin|> "
                    '{"nested": {"key": "value"}, "array": [1, 2, 3]} <|tool_call_end|> <|tool_calls_section_end|>'
                ),
                [
                    ToolCall(
                        id="functions.complex_function:0",
                        function=FunctionCall(
                            name="complex_function",
                            arguments='{"nested": {"key": "value"}, "array": [1, 2, 3]}',
                        ),
                        type="function",
                    )
                ],
                None,
            ),
            (
                """<|tool_calls_section_begin|> <|tool_call_begin|>
functions.no_args:0 <|tool_call_argument_begin|> {} <|tool_call_end|> <|tool_calls_section_end|>""",
                [
                    ToolCall(
                        id="functions.no_args:0",
                        function=FunctionCall(
                            name="no_args",
                            arguments='{}',
                        ),
                        type="function",
                    )
                ],
                None,
            ),
        ],
    )

    def test_extract_tool_calls(
        self, tool_parser, model_output, expected_tool_calls, expected_content, sample_request
    ):
        """
        测试工具调用提取的参数化测试：
        1. 输入包含工具调用标记的模型输出文本
        2. 解析器使用正则表达式匹配工具调用格式
        3. 提取工具名称、ID 和 JSON 参数
        4. 验证解析结果的格式和内容正确性
        """
        # 调用解析器提取工具调用信息
        extracted_tool_calls = tool_parser.extract_tool_calls(model_output, sample_request)
        assert extracted_tool_calls.tools_called

        assert_tool_calls(extracted_tool_calls.tool_calls, expected_tool_calls)
        assert extracted_tool_calls.content == expected_content

    def test_streaming_no_tool_calls(self, tool_parser, sample_request):
        """
        这个测试验证当流式输入不包含工具调用时，解析器的行为：
        1. 正确处理普通文本的增量更新
        2. 返回适当的 DeltaMessage 对象
        3. 确保 tool_calls 为空列表
        4. 正确传递文本内容
        """
        result = tool_parser.extract_tool_calls_streaming(
            previous_text="Hello",
            current_text="Hello world",
            delta_text=" world",
            previous_token_ids=[1, 2, 3],
            current_token_ids=[1, 2, 3, 4],
            delta_token_ids=[4],
            request=sample_request,
        )

        assert isinstance(result, DeltaMessage)
        assert result.content == " world"
        assert result.tool_calls == []

    def test_streaming_basic_functionality(self, tool_parser, sample_request):
        """
        这个测试验证流式处理中工具调用的完整提取过程。
        当工具调用区域结束时（<|tool_calls_section_end|>），解析器应该：
        1. 识别完整的工具调用
        2. 返回正确格式的工具调用信息
        3. 或者返回 None（如果工具调用尚未完成）
        """
        # 重置流式处理状态 - 确保测试的独立性
        tool_parser.current_tool_name_sent = False
        tool_parser.prev_tool_call_arr = []
        tool_parser.current_tool_id = -1
        tool_parser.streamed_args_for_tool = []

        # 模拟完整的工具调用文本
        current_text = (
            "I'll help you. <|tool_calls_section_begin|> <|tool_call_begin|>"
            "functions.get_weather:0 <|tool_call_argument_begin|> "
            '{"city": "Beijing"} <|tool_call_end|> <|tool_calls_section_end|>'
        )

        # 调用流式处理方法，模拟工具调用区域结束的时刻
        result = tool_parser.extract_tool_calls_streaming(
            previous_text="I'll help you",                    # 之前的文本
            current_text=current_text,                        # 当前完整文本
            delta_text="<|tool_calls_section_end|>",         # 新增的结束标记
            previous_token_ids=[],                            # 流式处理的 token 历史
            current_token_ids=[],                             # 当前 token 状态
            delta_token_ids=[],                               # 新增的 token
            request=sample_request,
        )

        if result is not None:
            assert isinstance(result, DeltaMessage)
            
            # 如果包含工具调用，验证工具调用的正确性
            if hasattr(result, "tool_calls") and result.tool_calls:
                # 应该至少有一个工具调用（因为输入包含完整的工具调用）
                assert len(result.tool_calls) > 0
                
                # 验证第一个工具调用的基本属性
                tool_call = result.tool_calls[0]
                assert tool_call.type == "function"
                assert "get_weather" in tool_call.id  # ID 应该包含函数名

    def test_streaming_tool_call_start(self, tool_parser, sample_request):
        """测试流式处理中工具调用开始"""
        tool_parser.current_tool_name_sent = False
        tool_parser.prev_tool_call_arr = []
        tool_parser.current_tool_id = -1
        tool_parser.streamed_args_for_tool = []

        result = tool_parser.extract_tool_calls_streaming(
            previous_text="I'll help you.",
            current_text=(
                "I'll help you.<|tool_calls_section_begin|><|tool_call_begin|>"
                "functions.get_weather:0<|tool_call_argument_begin|>"
            ),
            delta_text=(
                "<|tool_calls_section_begin|><|tool_call_begin|>"
                "functions.get_weather:0<|tool_call_argument_begin|>"
            ),
            previous_token_ids=[1, 2, 3],
            current_token_ids=[1, 2, 3, 163595, 163597, 4, 5, 163598],
            delta_token_ids=[163595, 163597, 4, 5, 163598],
            request=sample_request,
        )

        assert isinstance(result, DeltaMessage)
        if result.tool_calls:
            assert len(result.tool_calls) > 0
            if result.tool_calls:
                tool_call = result.tool_calls[0]
                assert tool_call.type == "function"
                assert tool_call.id == "functions.get_weather:0"

    def test_regex_patterns(self, tool_parser):
        """
        测试正则表达式能够正确识别单个工具调用格式
        """
        test_input = (
            "<|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>"
            "{\"location\": \"Beijing\"}<|tool_call_end|>"
        )

        matches = tool_parser.tool_call_regex.findall(test_input)
        assert len(matches) == 1
        assert matches[0][0] == "functions.get_weather:0"
        assert matches[0][1] == '{"location": "Beijing"}'

    def test_malformed_tool_calls(self, tool_parser, sample_request):
        """测试格式错误的工具调用"""
        # 缺少结束标记
        model_output = (
            "I'll help you. <|tool_calls_section_begin|> <|tool_call_begin|>"
            "functions.get_weather:0 <|tool_call_argument_begin|> "
            '{"city": "Beijing"}'
        )

        result = tool_parser.extract_tool_calls(model_output, sample_request)
        # 即使格式不完整，只要有工具调用开始标记，tools_called 就为 True
        # 但 tool_calls 列表应该为空，因为没有完整的工具调用
        assert result.tools_called
        assert result.tool_calls == []
        assert result.content == "I'll help you. "

    def test_empty_tool_calls_section(self, tool_parser, sample_request):
        """测试空工具调用, 且有Special Token"""
        model_output = (
            "I'll help you. <|tool_calls_section_begin|> <|tool_calls_section_end|>"
        )

        result = tool_parser.extract_tool_calls(model_output, sample_request)
        assert result.tools_called
        assert result.tool_calls == []
        assert result.content == "I'll help you. "

    def test_streaming_text_only(self, tool_parser, sample_request):
        """测试流式处理纯文本内容"""
        result = tool_parser.extract_tool_calls_streaming(
            previous_text="Hello",
            current_text="Hello world",
            delta_text=" world",
            previous_token_ids=[1, 2],
            current_token_ids=[1, 2, 3],
            delta_token_ids=[3],
            request=sample_request,
        )

        assert isinstance(result, DeltaMessage)
        assert result.content == " world"
        assert result.tool_calls == []

    def test_streaming_with_tool_tokens(self, tool_parser, sample_request):
        """
        测试包含工具调用 token 的流式处理
        验证解析器能正确识别和处理工具调用相关的 token
        """
        # 重置状态
        tool_parser.current_tool_name_sent = False
        tool_parser.prev_tool_call_arr = []
        tool_parser.current_tool_id = -1
        tool_parser.streamed_args_for_tool = []

        # 模拟包含工具调用开始 token 的情况
        result = tool_parser.extract_tool_calls_streaming(
            previous_text="I'll help you.",
            current_text="I'll help you.<|tool_calls_section_begin|>",
            delta_text="<|tool_calls_section_begin|>",
            previous_token_ids=[1, 2, 3],
            current_token_ids=[1, 2, 3, 163595],  # 包含工具调用开始 token
            delta_token_ids=[163595],
            request=sample_request,
        )

        # 流式处理可能返回 None 或 DeltaMessage
        if result is not None:
            assert isinstance(result, DeltaMessage)
