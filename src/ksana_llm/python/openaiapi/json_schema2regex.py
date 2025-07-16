# Copyright 2025 Tencent Inc.  All rights reserved.
# This file is try to add structured output prompt to the chat model.
# ==============================================================================

import json
from typing import Dict, Any


def create_structured_output_prompt(response_format: Dict[str, Any]) -> str:
    """
    创建结构化输出的提示文本
    
    Args:
        response_format: OpenAI 格式的 response_format 参数
        
    Returns:
        要添加到提示中的文本
    """
    if not response_format:
        return ""
    
    format_type = response_format.get("type")
    
    if format_type == "json_object":
        return "\n\nYou must respond with a valid JSON object. Start with { and end with }."
    
    elif format_type == "json_schema":
        json_schema_config = response_format.get("json_schema", {})
        schema = json_schema_config.get("schema") or json_schema_config.get("json_schema")
        
        if schema:
            # 生成示例，确保使用 ASCII 编码
            example = generate_json_example(schema)
            schema_str = json.dumps(schema, indent=2, ensure_ascii=True)
            example_str = json.dumps(example, indent=2, ensure_ascii=True)
            
            return f"""
You must respond with a valid JSON object that strictly follows this schema:

```json
{schema_str}
```

Example of a valid response:
```json
{example_str}
```

Remember: Your response must be a valid JSON object that matches the schema exactly."""
    
    return ""


def generate_json_example(schema: Dict[str, Any]) -> Any:
    """
    从 JSON Schema 生成示例数据
    
    Args:
        schema: JSON Schema 对象
        
    Returns:
        示例数据
    """
    schema_type = schema.get("type")
    
    if schema_type == "object":
        example = {}
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        
        # 添加所有必需属性和一些可选属性
        for prop_name, prop_schema in properties.items():
            if prop_name in required or len(example) < 3:
                example[prop_name] = generate_json_example(prop_schema)
        
        return example
    
    elif schema_type == "array":
        items = schema.get("items", {})
        min_items = schema.get("minItems", 0)
        
        # 生成最少数量的示例元素,至少会输出一个元素
        example = []
        for _ in range(max(1, min_items)):
            example.append(generate_json_example(items))
        
        return example
    
    elif schema_type == "string":
        if "enum" in schema:
            return schema["enum"][0]
        elif "format" in schema:
            format_examples = {
                "date": "2024-01-01",
                "time": "12:00:00",
                "date-time": "2024-01-01T12:00:00Z",
                "email": "user@example.com",
                "uri": "https://example.com",
                "uuid": "123e4567-e89b-12d3-a456-426614174000"
            }
            return format_examples.get(schema["format"], "example string")
        else:
            return "example string"
    
    elif schema_type == "number":
        if "enum" in schema:
            return schema["enum"][0]
        return 42.0
    
    elif schema_type == "integer":
        if "enum" in schema:
            return schema["enum"][0]
        return 42
    
    elif schema_type == "boolean":
        return True
    
    elif schema_type == "null":
        return None
    
    else:
        return ""
 