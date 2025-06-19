# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

import asyncio
import orjson
import aiohttp

API_URL = "http://localhost:8080/generate"
timeout = aiohttp.ClientTimeout(total=30*3600)
headers = {
    "User-Agent": "Benchmark Client",
    "Content-Type": "application/json",
}


async def send_req_sync(result_list, req_data):
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            output = None
            async with session.post(API_URL, headers=headers, data=req_data) as response:
                if response.status == 200:
                    chunk_acc = ""
                    async for chunk_bytes, _ in response.content.iter_chunks():
                        chunk_bytes = chunk_bytes.strip(b'\x00')
                        if not chunk_bytes:
                            continue
                        chunk = chunk_bytes.decode("utf-8")
                        if chunk_acc == "" and chunk.startswith("data: "):
                            chunk = chunk[len("data: "):]
                        if chunk == "[DONE]":
                            break
                        chunk_acc += chunk
                        try:
                            output = orjson.loads(chunk_acc)
                            chunk_acc = ""
                        except orjson.JSONDecodeError:
                            continue
            if (output is not None) and ("error" not in output):
                break

    output_text = output.get("texts", [""])[0].strip()
    result_list.append(output_text)


async def run_request(req_data_list):
    result_list = []
    for req_data in req_data_list:
        await send_req_sync(result_list, req_data)
    return result_list


def run(req_data_list):
    result_list = asyncio.run(run_request(req_data_list))
    return result_list


if __name__ == "__main__":
    input_data_list = [
        {	
            'prompt': '作为国际空间站上的宇航员，您意外地目睹了外星实体接近空间站。您如何向地面控制团队传达您的观察结果和建议？', 
            'sampling_config': {
                'temperature': 0.0, 
                'topk': 1, 
                'topp': 1.0, 
                'num_beams': 1, 
                'num_return_sequences': 1, 
                'length_penalty': 1.0, 
                'repetition_penalty': 1.0, 
                'no_repeat_ngram_size': 0, 
                'encoder_no_repeat_ngram_size': 0, 
                'logprobs': 0, 
                'max_new_tokens': 1024, 
                'ignore_eos': False
            }, 
            # [llama', 'llama-3', 'baichuan', 'qwen', 'vicuna', 'yi', 'chatglm']
            'model_type': 'qwen', 
            'use_chat_template': False, 
            'stream': True
        },
        {
            'prompt': '想象一下您是夏洛克·福尔摩斯，您被要求解开一个涉及失踪传家宝的谜团。请解释一下您找到该物品的策略。', 
            'sampling_config': {
                'temperature': 0.0, 
                'topk': 1, 
                'topp': 1.0, 
                'num_beams': 1, 
                'num_return_sequences': 1, 
                'length_penalty': 1.0, 
                'repetition_penalty': 1.0, 
                'no_repeat_ngram_size': 0, 
                'encoder_no_repeat_ngram_size': 0, 
                'logprobs': 0, 
                'max_new_tokens': 1024, 
                'ignore_eos': False
            }, 
            'stream': False, 
            # [llama', 'llama-3', 'baichuan', 'qwen', 'vicuna', 'yi', 'chatglm']
            'model_type': 'qwen', 
            'use_chat_template': False, 
            'stream': True
        }]
    req_data_list = [orjson.dumps(input_data) for input_data in input_data_list]
    result_list = run(req_data_list)
    print(f'result0: {result_list[0]}')
    print(f'result1: {result_list[1]}')
