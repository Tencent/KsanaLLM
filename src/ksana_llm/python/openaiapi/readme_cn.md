# KsanaLLM OpenAI API 兼容性使用指南

## 概述

KsanaLLM 提供了与 OpenAI API 完全兼容的接口，让您可以使用标准的 OpenAI 客户端库或 API 调用方式来访问 KsanaLLM 服务。

## 支持的 API 端点

### 1. Chat Completions API
- **端点**: `/v1/chat/completions`
- **方法**: `POST`
- **功能**: 生成聊天对话的回复，支持多轮对话、工具调用、流式输出、结构化响应等功能

### 2. Completions API (Legacy)
- **端点**: `/v1/completions`
- **方法**: `POST`
- **功能**: 传统的文本补全接口

### 3. Models API
- **端点**: `/models`
- **方法**: `GET`
- **功能**: 列出可用的模型

- **端点**: `/models/{model_id}`
- **方法**: `GET`
- **功能**: 获取特定模型的详细信息

### 4. 健康检查
- **端点**: `/health`
- **方法**: `GET`
- **功能**: 检查服务健康状态


## 待支持的接口：
embeddings API


## Quick Start

### 1. 启动服务

```bash
#打开 server 文件夹
cd KsanaLLM/src/ksana_llm/python

python serving_server.py \
    --config_file ${GIT_PROJECT_REPO_ROOT}/examples/ksana_llm.yaml  \
    --host 0.0.0.0 \
    --port 8080

#如果需要解析工具调用，请根据模型添加：
--enable-auto-tool-choice  --tool-call-parser deepseek_v3 \ 
# 如果需要解析推理模型的推理内容，请根据模型添加：
--reasoning-parser qwen3 \
# 如果需要使用固定的 chat-template 请根据 chat-template添加：
--chat-template

# 以DeepSeek_R1为例：
python serving_server.py \
--config_file ${GIT_PROJECT_REPO_ROOT}/examples/deepseek_fp8_perf.yaml \
--port 8080 \
--enable-auto-tool-choice     --tool-call-parser deepseek_v3  \
--reasoning-parser  deepseek_r1 \
--chat-template /workspace/KsanaLLM/examples/chat_templates/tool_chat_template_deepseekr1.jinja

本项目使用的 chat-template源自vLLM 项目中的开源模板，感谢 vLLM 项目组为开源做出的贡献
Ref:
https://github.com/vllm-project/vllm/blob/main/examples/tool_chat_template_deepseekv3.jinja

```

### 2. 使用 OpenAI Python 客户端发送请求

```python
from openai import OpenAI

# 配置客户端指向 KsanaLLM 服务
client = OpenAI(
    api_key="dummy-key",  # KsanaLLM 不需要 API key，但客户端需要这个参数
    base_url="http://localhost:8080/v1"
)

# 创建聊天完成请求
response = client.chat.completions.create(
    model="ksana-llm",  # 或使用您加载的具体模型名称
    messages=[
        {"role": "system", "content": "你是一个有帮助的助手。"},
        {"role": "user", "content": "你好，请介绍一下你自己。"}
    ],
    max_tokens=1000
)

print(response.choices[0].message.content)
```

### 3. 使用 cURL 直接调用

```bash
curl -X POST 'http://localhost:8080/v1/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer ksana-llm' \
-d '{
  "model": "ksana-llm",
  "messages": [
    {"role": "user", "content": "Hello, how are you?"}
  ]
}'

```
### 参数设置
在 KsanaLLM 中，采样参数需要和 do_sample 一起使用，只有 do_sample 为 True 时，且 topk 不为 1 时，采样参数才会正常改变，否则采样参数为系统设定的默认值

```bash
curl -X POST 'http://localhost:8080/v1/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer ksana-llm' \
-d '{
  "model": "ksana-llm",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
    {"role": "user", "content": "What is its population?"}
  ],
  "do_sample" : "True",
  "top_k" : "20",
  "temperature": "0.7"
}'

```


### 工具调用 (Function Calling)

KsanaLLM 支持 OpenAI API的工具调用功能：

```shell

curl -X POST "http://localhost:8080/v1/chat/completions" \
  -H "Content-Type: application/json" \
-d '{
  "model": "ksana-llm",
  "messages": [
    {
      "role": "user",
      "content": "What is the weather like in Boston today?"
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "The city and state, e.g. San Francisco, CA"
            },
            "unit": {
              "type": "string",
              "enum": ["celsius", "fahrenheit"]
            }
          },
          "required": ["location"]
        }
      }
    }
  ],
  "tool_choice": "auto"
}'

```

tool_choice 为 required 时，需要约束解码的限制，在 json_schema 开发完成后会马上支持

## 其他功能

### 1. 推理内容支持

KsanaLLM 支持返回模型的推理过程：

```shell

curl -X POST "http://localhost:8080/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ksana-llm",
    "messages": [
      {
        "role": "developer",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Hello!"
      }
    ]
  }'


```

### 2. 结构化输出

目前，KsanaLLM 支持较为简单的json_object的结构化输出功能，可以通过 prompt 和 response_format 来指定 json_object的返回
```shell

curl -X POST 'http://localhost:8080/v1/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer ksana-llm' \
-d '{
  "model": "ksana-llm",
  "messages": [
    {"role": "user", "content": "Which is the longest river in the world? The Nile River."}
  ],
  "response_format" : {
    "type": "json_object"
  }
}'

```


JSON Schema 格式的结构化输出正在规划开发中

## 配置选项

### 服务启动参数

- `--host`: 服务监听地址
- `--port`: 服务端口
- `--config_file`: YAML 配置文件路径，配置模型路径等参数
- `--tool_call_parser`: 工具调用解析器类型
- `--reasoning_parser`: 推理内容解析器类型
- `--enable_auto_tool_choice`: 是否启用自动工具选择
- `--chat-template`: 指定模型对话时使用的 chat_template
