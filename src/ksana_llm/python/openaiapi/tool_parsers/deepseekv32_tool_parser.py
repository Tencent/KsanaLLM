# Adapted from sglang:
# https://github.com/sgl-project/sglang/blob/v0.5.6.post1/python/sglang/srt/function_call/deepseekv32_detector.py
import json
import re

from collections.abc import Sequence
from typing import Union

from openaiapi.openai_protocol import (ChatCompletionRequest,
                                       DeltaFunctionCall, DeltaMessage,
                                       DeltaToolCall,
                                       ExtractedToolCallInformation,
                                       FunctionCall, ToolCall)
from openaiapi.tool_parsers.abstract_tool_parser import (
    ToolParser, ToolParserManager)

from openaiapi.transformers_utils.chat_utils import (AnyTokenizer,
                                                     make_tool_call_id)

from utilize.logger import get_logger

logger = get_logger(__name__)


@ToolParserManager.register_module("deepseek_v32")
class DeepSeekV32ToolParser(ToolParser):
    """
    Detector for DeepSeek V3.2 model function call format.

    The DeepSeek V3.2 format uses XML-like DSML tags to delimit function calls.
    Supports two parameter formats:

    Format 1 - XML Parameter Tags:
    ```
    <｜DSML｜function_calls>
        <｜DSML｜invoke name="function_name">
        <｜DSML｜parameter name="param_name" string="true">value</｜DSML｜parameter>
        ...
    </｜DSML｜invoke>
    </｜DSML｜function_calls>
    ```

    Format 2 - Direct JSON:
    ```
    <｜DSML｜function_calls>
        <｜DSML｜invoke name="function_name">
        {
            "param_name": "value"
        }
    </｜DSML｜invoke>
    </｜DSML｜function_calls>
    ```

    Examples:
    ```
    <｜DSML｜function_calls>
        <｜DSML｜invoke name="get_favorite_tourist_spot">
        <｜DSML｜parameter name="city" string="true">San Francisco</｜DSML｜parameter>
    </｜DSML｜invoke>
    </｜DSML｜function_calls>

    <｜DSML｜function_calls>
        <｜DSML｜invoke name="get_favorite_tourist_spot">
        { "city": "San Francisco" }
    </｜DSML｜invoke>
    </｜DSML｜function_calls>
    ```

    Key Components:
    - Tool Calls Section: Wrapped between `<｜DSML｜function_calls>` and `</｜DSML｜function_calls>`
    - Individual Tool Call: Wrapped between `<｜DSML｜invoke name="...">` and `</｜DSML｜invoke>`
    - Parameters: Either XML tags or direct JSON format
    - Supports multiple tool calls

    Reference: DeepSeek V3.2 format specification
    """

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)
        self.bot_token = "<｜DSML｜function_calls>"
        self.eot_token = "</｜DSML｜function_calls>"
        self.invoke_begin_regex = r'<｜DSML｜invoke\s+name="([^"]+)"\s*>'
        self.invoke_end_token = "</｜DSML｜invoke>"
        self.parameter_regex = r'<｜DSML｜parameter\s+name="([^"]+)"\s+string="([^"]+)"\s*>(.*?)</｜DSML｜parameter>'
        self._last_arguments = ""
        self.current_tool_id = -1
        self._buffer = ""

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a deepseek v32 format tool call."""
        return self.bot_token in text

    def _parse_parameters_from_xml(self, invoke_content: str) -> dict:
        """
        Parse parameters from either XML-like format or JSON format to dict.

        Supports two formats:
        1. XML parameter tags: <｜DSML｜parameter name="..." string="...">value</｜DSML｜parameter>
        2. Direct JSON: { "key": "value" }
        """
        # First, try to parse as direct JSON (new format)
        invoke_content_stripped = invoke_content.strip()

        if invoke_content_stripped.startswith("{") and invoke_content_stripped.endswith(
            "}"
        ):
            try:
                parameters = json.loads(invoke_content_stripped)
                if isinstance(parameters, dict):
                    return parameters
            except (json.JSONDecodeError, ValueError):
                # If JSON parsing fails, fall through to XML parsing
                pass

        # Fall back to XML parameter tag parsing (original format)
        parameters = {}
        param_matches = re.findall(self.parameter_regex, invoke_content, re.DOTALL)
        for param_name, param_type, param_value in param_matches:
            # Convert value based on type
            if param_type == "true":  # string type
                parameters[param_name] = param_value.strip()
            else:
                # Try to parse as JSON for other types
                try:
                    parameters[param_name] = json.loads(param_value.strip())
                except (json.JSONDecodeError, ValueError):
                    parameters[param_name] = param_value.strip()
        return parameters

    def extract_tool_calls(
        self, 
        model_output: str, 
        request: ChatCompletionRequest
    ) -> ExtractedToolCallInformation:
        """
        One-time parsing: Detects and parses tool calls in the provided model_output.

        :param model_output: The complete model_output to parse.
        :param request: The chat completion request containing available tools.
        :return: ParseResult indicating success or failure, consumed text, leftover text, and parsed calls.
        """
        logger.debug(f"model_output: {model_output}")
        idx = model_output.find(self.bot_token)
        normal_text = model_output[:idx].strip() if idx != -1 else model_output
        if self.bot_token not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=normal_text
            )

        tool_calls = []
        try:
            # Extract content between function_calls tags
            function_calls_match = re.search(
                r"<｜DSML｜function_calls>(.*?)</｜DSML｜function_calls>",
                model_output,
                re.DOTALL,
            )
            if not function_calls_match:
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=normal_text
                )

            function_calls_content = function_calls_match.group(1)
            # Find all invoke blocks
            invoke_pattern = (
                r'<｜DSML｜invoke\s+name="([^"]+)"\s*>(.*?)</｜DSML｜invoke>'
            )
            invoke_matches = re.findall(
                invoke_pattern, function_calls_content, re.DOTALL
            )

            for func_name, invoke_content in invoke_matches:
                # Parse parameters from XML format
                func_args = self._parse_parameters_from_xml(invoke_content)
                # TODO(winminkong): Add a switch to control whether to forward unknown tools.
                tool_calls.append(
                    ToolCall(
                        type="function",
                        function=FunctionCall(
                            name=func_name, arguments=json.dumps(func_args, ensure_ascii=False)
                        ),
                    )
                )
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=normal_text if normal_text else None,
            )
        except (IndexError, ValueError, AttributeError, TypeError):
            logger.error("Error in extracting tool call from response")
            # return the normal text if parsing fails
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> Union[DeltaMessage, None]:
        """
        Streaming incremental parsing tool calls for DeepSeekV32 format.
        Supports multiple consecutive invoke blocks.
        """
        logger.debug(f"previous_text: {previous_text}")
        logger.debug(f"delta_text: {delta_text}")
        self._buffer += delta_text
        current_text = self._buffer

        # Check if we have a tool call or any DSML-related content
        # Key insight: DSML tags contain distinctive markers like "｜DSML｜"
        # If we see these markers anywhere, we should keep buffering
        has_tool_call = (
            self.bot_token in current_text or "<｜DSML｜invoke" in current_text
        )
        # Check if buffer contains any DSML markers or ends with potential tag prefix
        # This handles partial/streaming DSML content
        dsml_markers = ["｜DSML｜", "<｜", "</｜"]
        potentially_dsml = any(marker in current_text for marker in dsml_markers)
        # Also check if text ends with start of a tag (to handle "<" arriving separately)
        dsml_prefixes = ["<", "<｜", "</", "</｜"]
        ends_with_prefix = any(
            current_text.rstrip().endswith(prefix) for prefix in dsml_prefixes
        )

        if not has_tool_call and not potentially_dsml and not ends_with_prefix:
            self._buffer = ""
            for e_token in [self.eot_token, self.invoke_end_token]:
                if e_token in delta_text:
                    delta_text = delta_text.replace(e_token, "")
            return DeltaMessage(content=delta_text)

        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(request.tools)

        all_calls: list[DeltaToolCall] = []
        try:
            # Loop to handle multiple consecutive invoke blocks
            while True:
                # Try to match an invoke block (may be partial)
                invoke_match = re.search(
                    pattern=r'<｜DSML｜invoke\s+name="([^"]+)"\s*>(.*?)(</｜DSML｜invoke>|$)',
                    string=current_text,
                    flags=re.DOTALL,
                )

                if not invoke_match:
                    break

                func_name = invoke_match.group(1).strip()
                invoke_content = invoke_match.group(2)
                # group(3) is either "</｜DSML｜invoke>" (complete) or "" (incomplete, matched with $)
                is_tool_end = bool(invoke_match.group(3))

                # Initialize state if this is the first tool call
                if self.current_tool_id == -1:
                    self.current_tool_id = 0
                    self.prev_tool_call_arr = []
                    self.streamed_args_for_tool = [""]

                # Don't pre-allocate arrays until we actually complete a tool call
                # This prevents _check_for_unstreamed_tool_args from sending incomplete calls

                # Parse current parameters from XML/JSON
                current_params = self._parse_parameters_from_xml(invoke_content)
                current_args_json = json.dumps(current_params, ensure_ascii=False)

                if func_name and not self.current_tool_name_sent:
                    calls_for_this_invoke: list[DeltaToolCall] = []
                    # Send tool name
                    calls_for_this_invoke.append(
                        DeltaToolCall(
                            index=self.current_tool_id,
                            id=make_tool_call_id(),
                            function=DeltaFunctionCall(
                                name=func_name,
                                arguments="",
                            ).model_dump(exclude_none=True),
                        ) 
                    )
                    self.current_tool_name_sent = True
                    all_calls.extend(calls_for_this_invoke)
                # Check if tool call is complete (has closing tag)
                if is_tool_end:
                    # Only emit the tool call when it's complete (saw </｜DSML｜invoke>)
                    # This ensures each function returns at most once
                    calls_for_this_invoke: list[DeltaToolCall] = []

                    # Note: invoke_content can be empty for functions with no parameters
                    # This is valid and should NOT be skipped

                    # Send parameters as complete JSON
                    # Always send parameters, even if empty, to maintain consistency
                    calls_for_this_invoke.append(
                        DeltaToolCall(
                            index=self.current_tool_id,
                            function=DeltaFunctionCall(
                                arguments=current_args_json
                            ).model_dump(exclude_none=True),
                        )
                    )

                    # Ensure arrays are large enough for current tool
                    while len(self.prev_tool_call_arr) <= self.current_tool_id:
                        self.prev_tool_call_arr.append({})
                    while len(self.streamed_args_for_tool) <= self.current_tool_id:
                        self.streamed_args_for_tool.append("")

                    # Update the stored arguments
                    self.prev_tool_call_arr[self.current_tool_id] = {
                        "name": func_name,
                        "arguments": current_params,
                    }
                    self.streamed_args_for_tool[self.current_tool_id] = (
                        current_args_json
                    )

                    # Remove the completed tool call from buffer
                    self._buffer = current_text[invoke_match.end() :]
                    current_text = self._buffer  # Update for next iteration

                    # Add calls for this invoke to all_calls
                    all_calls.extend(calls_for_this_invoke)

                    # Move to next tool call
                    self.current_tool_id += 1
                    self._last_arguments = ""
                    self.current_tool_name_sent = False

                    # Don't pre-allocate arrays for the next tool
                    # Only allocate when we actually complete a tool call
                    # This prevents _check_for_unstreamed_tool_args from sending incomplete calls

                    # Continue loop to check for more invoke blocks
                    continue
                else:
                    # Tool call not complete yet, don't return anything
                    # Wait for more chunks until we see </｜DSML｜invoke>
                    break

            # No more invoke blocks found
            if all_calls:
                return DeltaMessage(tool_calls=all_calls)
            else:
                return None

        except (IndexError, ValueError, AttributeError, TypeError, KeyError):
            logger.error("Error trying to handle streaming tool call")
            return DeltaMessage(content=current_text)

    # TODO(winminkong): Add structure tags
    # def structure_info(self) -> _GetInfoFunc:
    #     return lambda name: StructureInfo(
    #         begin=f'<｜DSML｜invoke name="{name}">',
    #         end="</｜DSML｜invoke>",
    #         trigger=f'<｜DSML｜invoke name="{name}">',
    #     )
