# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ==============================================================================
# Adapted from vLLM project
# [vLLM Project]
# Ref:
# https://github.com/vllm-project/vllm/blob/v0.9.1/vllm/entrypoints/openai/serving_models.py
# ==============================================================================

"""
OpenAI Models API服务实现
参考OpenAI官方API格式，确保完全兼容
"""

import time
from http import HTTPStatus
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, ConfigDict
from fastapi.responses import JSONResponse

from openaiapi.openai_adapter import OpenAIConfig, ErrorType, KsanaOpenAIServing

from utilize.logger import get_logger

logger = get_logger(__name__)


class ModelObject(BaseModel):
    """OpenAI Model对象 - 完全符合官方格式"""
    model_config = ConfigDict(exclude_none=True)
    
    id: str = Field(..., description="模型ID")
    object: str = Field(default="model", description="对象类型，固定为'model'")
    created: int = Field(..., description="创建时间戳")
    owned_by: str = Field(..., description="模型所有者")
    permission: List[Dict[str, Any]] = Field(default_factory=list, description="权限列表")
    root: Optional[str] = Field(default=None, description="根模型")
    parent: Optional[str] = Field(default=None, description="父模型")


class ModelsListResponse(BaseModel):
    """Models列表响应 - 完全符合OpenAI格式"""
    model_config = ConfigDict(exclude_none=True)
    
    object: str = Field(default="list", description="对象类型，固定为'list'")
    data: List[ModelObject] = Field(..., description="模型列表")


class KsanaOpenAIServingModels(KsanaOpenAIServing):
    """
    KsanaLLM Models API实现
    完全符合OpenAI官方API格式
    """
    
    def __init__(self, llm_server, config: Optional[OpenAIConfig] = None):
        super().__init__(llm_server, config)
        self._models_cache = None
        self._cache_timestamp = 0
        self._cache_ttl = 300  # 缓存5分钟
    
    def _get_model_permissions(self) -> List[Dict[str, Any]]:
        """获取模型权限 - 符合OpenAI格式"""
        # 只包含值为 True 或有意义的字段
        return [
            {
                "id": "modelperm-ksana",
                "object": "model_permission",
                "created": int(time.time()),
                "allow_sampling": True,
                "allow_logprobs": True,
                "allow_view": True,
                "organization": "*"
            }
        ]
    
    def _extract_detailed_model_info(self) -> ModelObject:
        """提取详细的模型信息 - 完全符合OpenAI格式"""
        try:
            engine_args = getattr(self.llm_server, 'engine_args', None)
            model_dir = getattr(engine_args, 'model_dir', None) if engine_args else None
            
            if model_dir:
                model_name = model_dir.rstrip('/').split('/')[-1]
                if model_name.startswith('./'):
                    model_name = model_name[2:]
            else:
                model_name = self.config.default_model_name
            
            model_info = {
                "id": model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "ksana-llm",
                "permission": self._get_model_permissions(),
                "root": model_name,
                "parent": None,
            }
            
            return ModelObject(**model_info)
            
        except (AttributeError, TypeError, ValueError) as e:
            logger.warning(f"Failed to extract detailed model info: {e}")
            return self._get_default_model_object()
    
    def _get_default_model_object(self) -> ModelObject:
        """获取默认模型对象 - 完全符合OpenAI格式"""
        return ModelObject(
            id=self.config.default_model_name,
            object="model",
            created=int(time.time()),
            owned_by="ksana-llm",
            permission=self._get_model_permissions(),
            root=self.config.default_model_name,
            parent=None
        )
    
    def _get_available_models(self) -> List[ModelObject]:
        """获取可用模型列表"""
        # 检查缓存
        current_time = time.time()
        if (self._models_cache is not None and current_time - self._cache_timestamp < self._cache_ttl):
            return self._models_cache
        
        try:
            # 获取主模型
            main_model = self._extract_detailed_model_info()
            models = [main_model]
            
            # TODO: 如果支持多模型，可以在这里添加其他模型
            # 目前只返回主模型            
            # 更新缓存
            self._models_cache = models
            self._cache_timestamp = current_time
            
            return models
            
        except (AttributeError, TypeError, ValueError) as e:
            logger.error(f"Failed to get available models: {e}")
            # 返回默认模型
            default_model = self._get_default_model_object()
            return [default_model]
    
    def list_models(self) -> JSONResponse:
        """
        列出所有可用模型
        GET /v1/models
        完全符合OpenAI API格式
        """
        models = self._get_available_models()
        
        response = ModelsListResponse(
            object="list",
            data=models
        )
        
        return JSONResponse(content=response.model_dump(exclude_none=True))
    
    def get_model(self, model_id: str) -> JSONResponse:
        """
        获取特定模型信息
        GET /v1/models/{model_id}
        完全符合OpenAI API格式
        """
        models = self._get_available_models()
        
        # 查找指定的模型
        target_model = None
        for model in models:
            if model.id == model_id:
                target_model = model
                break
        
        if target_model is None:
            return self.create_error_response(
                f"The model '{model_id}' does not exist",
                ErrorType.NOT_FOUND_ERROR,
                HTTPStatus.NOT_FOUND
            )
        
        return JSONResponse(content=target_model.model_dump(exclude_none=True))
            
    
    def delete_model(self, model_id: str) -> JSONResponse:
        """
        删除模型 (通常用于fine-tuned模型)
        DELETE /v1/models/{model_id}
        完全符合OpenAI API格式
        """
        # TODO（ethanyczeng): 实现模型删除逻辑
        # 目前返回不支持的错误
        return self.create_error_response(
            f"Model deletion is not supported for model '{model_id}'",
            ErrorType.BAD_REQUEST_ERROR,
            HTTPStatus.BAD_REQUEST
        )
    
    def invalidate_cache(self):
        """清除模型缓存"""
        self._models_cache = None
        self._cache_timestamp = 0
        logger.info("Models cache invalidated")
    
    def is_base_model(self, model_name: str) -> bool:
        """检查是否是基础模型"""
        if not model_name:
            return True
        
        # 获取可用模型列表
        models = self._get_available_models()
        
        # 检查模型名称是否匹配
        for model in models:
            if model.id == model_name:
                return True
        
        # 检查是否是默认模型名称
        if model_name == self.config.default_model_name:
            return True
        
        return False
