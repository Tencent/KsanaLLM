# Forward接口C++客户端示例

## Forward接口说明

请求模型对输入的提示词或tokens执行单步推理，即context阶段。
用户可以指定若干请求目标，并指定token归约模式，包括transformer，layernorm和logits对应的推理结果。
具体而言，对于transformer或layernorm目标，支持"GATHER_ALL"归约模式，分别用于获取进行一次推理后的transformer或layernorm结果。
对于logits目标，支持"GATHER_ALL"或"GATHER_TOKEN_ID"，分别用于获取推理后的所需位置所有token的logits结果（softmax处理前）或所需位置对应token的logits结果（softmax处理后）。

该接口支持批处理，用户可以一次性提交一组请求，之后会返回与这组请求对应的一组响应。

比如以下请求获取模型对"hello world"进行一次推理后的所需位置对应token的计算结果，即对除首token外的各个token的预测概率（已经过softmax处理），后续可以进一步用于计算困惑度（PPL）等。

```bash
# the user request
{
    "requests": [
        {
            "prompt": "hello world" # the input prompt
            "request_target": [
                {
                    "target_name": "logits", # request for logits result
                    "slice_pos": [[0, -2]], # specify the interval of the result (negative indices count from the end)
                    "token_reduce_mode": "GATHER_TOKEN_ID", # the redude mode
                }, # you can append more targets
            ]
        }, # you can append more requests
    ]
}
# the service response
{
    "responses": [
        {
            "input_token_ids": [14990, 1879], # the tokens of "hello world"
            "response": [
                {
                    "target_name": "logits",
                    "tensor":
                        {
                            "data": b"oisyOw==", # the result is encoded into base64 [0.00271867]
                            "shape": [1],
                            "dtype": "float32",
                        }
                }
            ]
        }
    ]
}
```

具体的请求格式请参见`src/ksana_llm/utils/request_serial.h`。
C++用户可以仿照该文件的写法，提前定义好与接口对应的结构体（只需要定义自己要用的字段即可）。
每次请求时赋值请求结构体，打包为msgpack格式的二进制发送，再将返回解包为响应结构体，处理结果即可。

## 第三方库依赖

- [httplib](https://github.com/yhirose/cpp-httplib)：选择任意可以发送HTTP请求的客户端

- [msgpack-c](https://github.com/msgpack/msgpack-c/tree/cpp_master)：将HTTP请求体打包为msgpack格式，并对msgpack格式的HTTP响应体解包

- [base64](https://github.com/tobiaslocker/base64)：用base64解码返回的数值结果，低于C++17的可以使用[base64-decode-snippet-in-c](https://stackoverflow.com/questions/180947/base64-decode-snippet-in-c)

## 独立编译与使用

```bash
# download the 3rdparty dependencies
git clone --depth 1 https://github.com/yhirose/cpp-httplib
git clone --depth 1 -b cpp_master https://github.com/msgpack/msgpack-c/
git clone --depth 1 https://github.com/tobiaslocker/base64
# compile
g++ serving_forward_client.cpp -Icpp-httplib/ -Imsgpack-c/include -Ibase64/include -I../../ -pthread -O2 -DMSGPACK_NO_BOOST -o serving_forward_client
# run
./serving_forward_client --host 127.0.0.1 --port 8888 --api forward
"
input_token_ids : [ 9707 11 1879 0 ], target_name : logits, tensor :
[ 0.15906 0.00106 ]
input_token_ids : [ 1 22 13 ], target_name : logits, tensor :
[ 0.00122 0.12967 ]
request duration: 16 ms
"
# help
./serving_forward_client --help
"
usage: ./serving_forward_client [-h, --help] [-s, --host HOST] [-p, --port PORT] [-a, --api API]
optional arguments:
  -h, --help       show this help message and exit
  -s, --host HOST  server host address
  -p, --port PORT  server port
  -a, --api  API   server api
"
```
