# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ==============================================================================
# Adapted from vLLM project
# [vLLM Project]

# Ref:
# https://github.com/vllm-project/vllm/blob/v0.9.1/vllm/entrypoints/openai/serving_embedding.py
# ==============================================================================


"""
OpenAI Embeddings API服务实现
"""

import uuid
from typing import List, Optional, Union
from http import HTTPStatus

from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict

from openaiapi.openai_adapter import KsanaOpenAIServing, ErrorType, OpenAIConfig
from openaiapi.request_converter import RequestConverter

from utilize.logger import get_logger

logger = get_logger(__name__)


class EmbeddingRequest(BaseModel):
    """Embedding请求模型"""
    input: Union[str, List[str]] = Field(..., description="输入文本")
    model: str = Field(default="ksana-llm", description="模型名称")
    encoding_format: Optional[str] = Field(default="float", description="编码格式")
    dimensions: Optional[int] = Field(default=None, description="向量维度")
    user: Optional[str] = Field(default=None, description="用户标识")


class EmbeddingData(BaseModel):
    """单个embedding数据"""
    model_config = ConfigDict(exclude_none=True)
    
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingUsage(BaseModel):
    """Embedding使用统计"""
    model_config = ConfigDict(exclude_none=True)
    
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    """Embedding响应模型"""
    model_config = ConfigDict(exclude_none=True)
    
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: EmbeddingUsage


class KsanaOpenAIServingEmbedding(KsanaOpenAIServing):
    """
    KsanaLLM Embeddings API实现
    """
    
    def __init__(self, llm_server, config: Optional[OpenAIConfig] = None):
        super().__init__(llm_server, config)
    
    async def create_embedding(self, request: Request) -> JSONResponse:
        """处理Embeddings API请求 - 使用标准化格式化器"""
        try:
            request_dict = await request.json()
            embedding_request = EmbeddingRequest(**request_dict)
            
            # 处理输入文本
            if isinstance(embedding_request.input, str):
                texts = [embedding_request.input]
            else:
                texts = embedding_request.input
            
            # 生成embeddings
            embeddings = []
            
            for text in texts:
                # TODO: 实现实际的embedding生成逻辑
                embedding_vector = await self._generate_embedding(text)
                embeddings.append(embedding_vector)
            
            # 使用标准化格式化器
            converter = RequestConverter(self.config, tokenizer=self.tokenizer)
            
            request_id = f"embd-{uuid.uuid4().hex}"
            model_name = self._get_model_name(embedding_request.model)
            
            response = converter.format_embedding_response(
                request_id=request_id,
                model_name=model_name,
                embeddings=embeddings,
                input_texts=texts,
                encoding_format=getattr(embedding_request, 'encoding_format', 'float')
            )
            
            return JSONResponse(content=response.model_dump(exclude_none=True))
            
        except (ValueError, TypeError, KeyError) as e:
            logger.error(f"Embeddings API validation error: {e}")
            return self.create_error_response(
                str(e),
                ErrorType.VALIDATION_ERROR,
                HTTPStatus.BAD_REQUEST
            )
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """生成embedding向量"""
        # TODO(ethanyczeng): 这里应该调用实际的embedding模型
        # 可能需要使用不同的模型或者调用外部embedding服务
        
        # 示例实现：返回固定维度的随机向量
        import random
        dimensions = 768  # 常见的embedding维度
        
        # 为了演示，这里生成一个简单的向量
        # 实际实现中，您需要：
        # 1. 加载embedding模型
        # 2. 对文本进行tokenization
        # 3. 通过模型生成embedding
        
        embedding = [random.uniform(-1, 1) for _ in range(dimensions)]
        return embedding
    
    def _count_tokens(self, text: str) -> int:
        """计算文本的token数量"""
        if hasattr(self.llm_server, 'tokenizer') and self.llm_server.tokenizer is not None:
            tokens = self.llm_server.tokenizer.encode(text)
            return len(tokens)
        
        # 简单估算：按空格分割
        return len(text.split())
