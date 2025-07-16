# Copyright 2025 Tencent Inc.  All rights reserved.
#
# ==============================================================================
"""
简单的 OpenAI API 调试测试脚本
"""

import os
import traceback
from openai import OpenAI
import openai

# 测试配置
API_BASE = os.getenv("KSANA_API_BASE", "http://localhost:8080/v1")
API_KEY = os.getenv("KSANA_API_KEY", "dummy-key")
MODEL_NAME = os.getenv("KSANA_MODEL_NAME", "ksana-llm")

# 创建客户端
client = OpenAI(api_key=API_KEY, base_url=API_BASE)


def test_basic_completion():
    """测试基本的聊天完成"""
    print("\n=== 测试基本聊天完成 ===")
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": "Hello, say hi back"}
            ],
            max_tokens=50,
            temperature=0.7
        )
        print(f"✅ 成功: {response.choices[0].message.content}")
        return True
    except (ValueError, TypeError, RuntimeError, ConnectionError) as e:
        print(f"❌ 失败: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


def test_with_system_message():
    """测试带系统消息的聊天"""
    print("\n=== 测试带系统消息 ===")
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2?"}
            ],
            max_tokens=50
        )
        print(f"✅ 成功: {response.choices[0].message.content}")
        return True
    except (ValueError, TypeError, RuntimeError, ConnectionError) as e:
        print(f"❌ 失败: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


def test_streaming():
    """测试流式响应"""
    print("\n=== 测试流式响应 ===")
    try:
        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": "Count from 1 to 3"}
            ],
            max_tokens=50,
            stream=True
        )
        
        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                print(f"收到 chunk: {chunk.choices[0].delta.content}")
        
        print(f"✅ 成功: 完整响应 = {full_response}")
        return True
    except (ValueError, TypeError, RuntimeError, ConnectionError) as e:
        print(f"❌ 失败: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


def test_empty_messages():
    """测试空消息列表（应该失败）"""
    print("\n=== 测试空消息列表 ===")
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[],
            max_tokens=50
        )
        print(f"❌ 不应该成功: {response}")
        return False
    except openai.BadRequestError as e:
        print(f"✅ 预期的失败: {type(e).__name__}: {e}")
        # 验证错误消息
        if "Messages cannot be empty" in str(e):
            print("✅ 错误消息正确")
        return True
    except (ValueError, TypeError, RuntimeError, ConnectionError) as e:
        print(f"✅ 预期的失败: {type(e).__name__}: {e}")
        return True


def test_invalid_model():
    """测试无效的模型名"""
    print("\n=== 测试无效模型名 ===")
    try:
        response = client.chat.completions.create(
            model="invalid-model-xyz",
            messages=[
                {"role": "user", "content": "Hello"}
            ],
            max_tokens=50
        )
        print(f"❌ 不应该成功: {response}")
        return False
    except openai.NotFoundError as e:
        print(f"✅ 预期的失败: {type(e).__name__}: {e}")
        return True
    except (ValueError, TypeError, RuntimeError, ConnectionError) as e:
        print(f"✅ 预期的失败: {type(e).__name__}: {e}")
        return True


def test_tool_calling():
    """测试工具调用功能"""
    print("\n=== 测试工具调用 ===")
    
    tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather in a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": "What's the weather in Beijing?"}
            ],
            tools=[tool],
            tool_choice="auto",
            max_tokens=100
        )
        
        if response.choices[0].message.tool_calls:
            print(f"✅ 成功: 调用了工具 {response.choices[0].message.tool_calls[0].function.name}")
        else:
            print(f"✅ 成功: 返回了文本响应 {response.choices[0].message.content}")
        return True
    except (ValueError, TypeError, RuntimeError, ConnectionError) as e:
        print(f"❌ 失败: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


def test_multiple_messages():
    """测试多轮对话"""
    print("\n=== 测试多轮对话 ===")
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": "My name is Alice"},
                {"role": "assistant", "content": "Nice to meet you, Alice!"},
                {"role": "user", "content": "What's my name?"}
            ],
            max_tokens=50
        )
        print(f"✅ 成功: {response.choices[0].message.content}")
        return True
    except (ValueError, TypeError, RuntimeError, ConnectionError) as e:
        print(f"❌ 失败: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print(f"OpenAI API 测试")
    print(f"API Base: {API_BASE}")
    print(f"Model: {MODEL_NAME}")
    
    tests = [
        test_basic_completion,
        test_with_system_message,
        test_streaming,
        test_multiple_messages,
        test_empty_messages,
        test_invalid_model,
        test_tool_calling,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except (KeyboardInterrupt, SystemExit):
            raise
        except (ValueError, TypeError, RuntimeError, ConnectionError) as e:
            print(f"测试异常: {e}")
            failed += 1
    
    print(f"\n=== 测试结果 ===")
    print(f"通过: {passed}")
    print(f"失败: {failed}")
    print(f"总计: {passed + failed}")


if __name__ == "__main__":
    main()
