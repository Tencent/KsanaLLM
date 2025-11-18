# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
from transformers import PreTrainedTokenizerBase

from openaiapi.reasoning import ReasoningParser, ReasoningParserManager

from utilize.logger import get_logger

logger = get_logger(__name__)


@ReasoningParserManager.register_module("deepseek_v3")
class DeepSeekV3ReasoningParser(ReasoningParser):
    """
    Reasoning parser for DeepSeek V3.1 model.

    The DeepSeek V3.1 model uses <think>...</think> tokens to denote reasoning
    text. This parser extracts the reasoning content from the model output.
    """

    # Token ID映射：将字符串token转换为对应的token ID用于高效处理
    think_start_token_id: int  # "<think>"对应的token ID
    think_end_token_id: int    # "</think>"对应的token ID

    # 推理标记的字符串表示
    think_start_token: str = "<think>"   # 推理开始标记
    think_end_token: str = "</think>"    # 推理结束标记
    
    # 特殊结束确认机制：
    # 因为模型在推理过程中可能会生成"</think>"作为推理内容的一部分，
    # 这会导致误判推理结束。因此需要特殊的确认机制。
    end_confirmation_token_id: int = 201  # 特殊结束token ID (通常是换行符\n的token ID)
                                         # 只有当</think>后紧跟此token时，才确认推理真正结束

    # 延迟确认状态管理：
    # 当遇到</think>时，不立即确认推理结束，而是等待下一个token来确认
    _awaiting_end_confirmation: bool = False    # 标记是否有待确认的</think>token
    _pending_reasoning_content: str = ""      # 暂存待确认的推理内容，如果确认失败需要作为推理内容输出

    # 推理状态缓存：
    # 一旦确认推理结束，缓存此状态避免重复计算
    _reasoning_ended: bool = False      # 标记推理是否已经确认结束，用于性能优化

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        super().__init__(tokenizer)

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ReasoningParser "
                "constructor during construction.")

        self.think_start_token_id = self.vocab.get(self.think_start_token)
        self.think_end_token_id = self.vocab.get(self.think_end_token)
        if self.think_start_token_id is None or self.think_end_token_id is None:
            raise RuntimeError(
                "DeepSeek V3.1 reasoning parser could not locate think start/end "
                "tokens in the tokenizer!")
        
        # Initialize state management
        self._awaiting_end_confirmation = False
        self._pending_reasoning_content = ""
        self._reasoning_ended = False

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        """
        判断推理是否真正结束
        
        核心逻辑：只有当</think>后紧跟end_confirmation_token_id(201)时，才确认推理结束
        
        Args:
            input_ids: 当前的token ID序列
            
        Returns:
            bool: True表示推理已确认结束，False表示推理仍在进行或未确认结束
            
        状态管理：
        - 使用_reasoning_ended缓存结果，避免重复计算
        - 一旦确认推理结束，后续调用直接返回True
        """
        # If reasoning has already been confirmed as ended, return True immediately
        if self._reasoning_ended:
            return True
            
        if self.think_end_token_id not in input_ids:
            return False
        
        # Find all occurrences of think_end_token_id
        for token_id in input_ids:
            if token_id == self.think_end_token_id:
                # NOTE(winminkong): There should not be a "\n" following "deepseek v3.1 </think>"
                self._reasoning_ended = True
                return True
        
        return False

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """
        提取结束token之后的内容token IDs
        
        Args:
            input_ids: 输入的token ID序列
            
        Returns:
            list[int]: 结束token之后的token ID列表
        """
        if self.think_end_token_id not in input_ids[:-1]:
            return []
        else:
            return input_ids[input_ids.index(self.think_end_token_id) + 1:]
