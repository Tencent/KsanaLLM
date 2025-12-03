# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

import sys
import argparse
import threading
import queue
import os
import subprocess

from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor

import torch

# Add paths for C++ lib python wrapper and python interface
sys.path.insert(0, '../../build/lib')
sys.path.insert(0, '../../src/ksana_llm/python')
sys.path.insert(0, '.')


# Define a function to parse command line arguments
def args_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file',
                        type=str,
                        default="examples/ksana_llm.yaml",
                        help='serving config file')
    parser.add_argument('--tokenizer_dir',
                        type=str,
                        default="/model/llama-hf/13B",
                        help='tokenizer dir')
    parser.add_argument('--model', type=str, default="qwen", help='model type')
    parser.add_argument("--distributed",
                        type=bool,
                        default=False,
                        help="enable the distributed mode")
    args = parser.parse_args()
    return args


def enqueue_output(file, queue):
    for line in iter(file.readline, ''):
        queue.put(line)
    file.close()


def read_popen_pipes(p):
    with ThreadPoolExecutor(2) as pool:
        q_stdout, q_stderr = Queue(), Queue()
        pool.submit(enqueue_output, p.stdout, q_stdout)
        pool.submit(enqueue_output, p.stderr, q_stderr)
        while True:
            if p.poll() is not None and q_stdout.empty() and q_stderr.empty():
                break
            out_line = err_line = ''
            try:
                out_line = q_stdout.get_nowait()
            except Empty:
                pass
            try:
                err_line = q_stderr.get_nowait()
            except Empty:
                pass
            yield (out_line, err_line)


def wait_for_server_launch(server_proc, server_status_queue):
    for _, err_line in read_popen_pipes(server_proc):
        if len(err_line) > 0:
            print(err_line, end="")
            server_status_queue.put_nowait(err_line)


def read_from_csv(csv_file, col_idx=0, remove_head=True):
    import csv
    csv_reader = csv.reader(open(csv_file))
    return [row[0] for row in csv_reader]


def get_ref_results(model):
    if model == "deepseek_v2":
        return [
            "作为夏洛克·福尔摩斯，我会采取以下策略来解开失踪传家宝的谜团：\n\n"
            "1. **收集信息**：首先，我会与失主进行详细的交谈，了解传家宝的详细信息，包括它的历史、价值、以及它在家族中的意义。"
            "同时，我还会收集所有可能的线索，比如失主最后一次看到传家宝的时间和地点，以及任何可能的目击者。\n\n"
            "2. **现场调查**：我会亲自前往失物地点进行现场调查，寻找任何可能的线索，比如指纹、脚印、遗失物品等。我还会观察周围环境，寻找任何异常或不寻常的地方。\n\n"
            "3. **逻辑推理**：我会运用我的逻辑推理能力，将收集到的信息和线索进行分析，尝试找出其中的关联和模式。我会考虑所有可能的情况，包括失主的动机、可能的嫌疑人，以及任何可能的隐藏地点。\n\n"
            "4. **科学检验**：如果需要，我会使用我的化学和物理知识，进行一些简单的科学检验，比如使用放大镜检查现场的微小痕迹，或者进行一些简单的化学测试，以帮助我更好地理解情况。\n\n"
            "5. **利用我的网络**：我会利用我在伦敦的广泛人脉和联系，寻找可能的线索或目击者。我可能会联系警察、侦探同行、甚至是黑社会成员，以获取他们可能知道的信息。\n\n"
            "6. **制定计划**：基于我的调查和推理，我会制定一个详细的计划，以找回失踪的传家宝。这可能包括秘密监视嫌疑人、设置陷阱捕捉小偷，或者在特定的时间和地点进行搜查。\n\n"
            "7. **行动**：最后，我会执行我的计划，追踪和找回失踪的传家宝。在这个过程中，我会保持警惕，随时准备应对可能出现的任何意外情况。\n\n"
            "通过这些策略，我相信我能够解开这个谜团，找回失踪的传家宝。",
            "在太空中，宇航员与地面控制团队之间的通信是通过无线电波进行的。如果宇航员在太空中观察到了不明飞行物或外星实体，他们应该立即使用国际空间站上的紧急通信系统，向地面控制中心报告他们的观察。\n\n"
            "报告应该包括以下几个关键信息：\n\n"
            "1. **实体的描述**：描述实体的大小、形状、颜色、运动方式等特征。\n"
            "2. **位置和时间**：提供实体出现的确切位置和时间，以及宇航员观察到它的具体时间。\n"
            "3. **宇航员的安全状况**：报告宇航员的身体状况和空间站的整体状况，确保地面团队了解宇航员的安全。\n"
            "4. **建议和请求**：提出宇航员认为可能需要的任何行动，例如，如果需要进行进一步的观察、拍摄或记录，或者如果需要采取安全措施。\n\n"
            "地面控制团队将根据宇航员的报告和当前的太空任务规则，评估情况并决定如何应对。他们可能会要求宇航员进行更详细的观察，或者根据情况采取预防措施，比如改变空间站的轨道或进行紧急撤离准备。\n\n"
            "重要的是，宇航员应该保持冷静，准确地传达信息，并遵循地面控制团队的指导。在太空中，任何异常情况都需要谨慎和专业的处理。"
        ]
    elif model == "qwen":
        return [
            "1. 确保信息准确：首先，我需要确保我所观察到的外星实体是真实的，而不是幻觉。我需要确保我所观察到的实体是外星生物，而不"
            "是其他任何可能的物体。\n\n2. 保持冷静：在面对未知的外星实体时，保持冷静是非常重要的。我需要保持冷静，以便能够有效地传"
            "达我的观察结果和建议。\n\n3. 保持开放和尊重的态度：我需要保持开放和尊重的态度，以便能够与地面控制团队进行有效的沟通。"
            "我需要尊重他们的观点和决定，同时也需要尊重他们的工作。\n\n4. 提供信息：我需要提供我所观察到的外星实体的信息，以便他们"
            "能够理解我的观察结果。我需要提供我所观察到的外星实体的特征，以及我所观察到的外星实体的行为。\n\n5. 提供建议：我需要提"
            "供我所观察到的外星实体的建议，以便他们能够更好地理解我的观察结果。我需要提供我所观察到的外星实体的可能的解决方案，以及"
            "我所观察到的外星实体可能遇到的问题。\n\n6. 保持沟通：我需要保持与地面控制团队的沟通，以便他们能够及时了解我的观察结果"
            "和建议。我需要保持与地面控制团队的沟通，以便他们能够及时了解我的观察结果和建议。\n\n7. 保持专业：我需要保持专业，以便"
            "能够有效地传达我的观察结果和建议。我需要保持专业，以便能够有效地传达我的观察结果和建议。",
            "首先，我会仔细阅读传家宝的描述，以了解其价值和用途。然后，我会仔细检查传家宝的标签和包装，以确保它没有被损坏或丢失。我"
            "会仔细检查传家宝的内部，以确保它没有被破坏或丢失。最后，我会仔细检查传家宝的标签，以确保它没有被修改或更改。\n\n如果"
            "传家宝的标签上没有明显的标记，我会尝试使用我的推理能力来推测其价值。例如，如果传家宝的标签上写着“价值100万英镑”，我可"
            "能会推测它可能是一个非常重要的物品，价值超过100万英镑。\n\n如果传家宝的标签上写着“价值100万英镑”，我可能会推测它可能"
            "是一个非常重要的物品，价值超过100万英镑。如果传家宝的标签上写着“价值100万英镑”，我可能会推测它可能是一个非常重要的物"
            "品，价值超过100万英镑。如果传家宝的标签上写着“价值100万英镑”，我可能会推测它可能是一个非常重要的物品，价值超过100万"
            "英镑。\n\n如果传家宝的标签上写着“价值100万英镑”，我可能会推测它可能是一个非常重要的物品，价值超过100万英镑。如果传家"
            "宝的标签上写着“价值100万英镑”，我可能会推测它可能是一个非常重要的物品，价值超过100万英镑。如果传家宝的标签上写着“价值"
            "100万英镑”，我可能会推测它可能是一个非常重要的物品，价值超过100万英镑。\n\n如果传家宝的标签上写着“价值100万英镑”，我"
            "可能会推测它可能是一个非常重要的物品，价值超过100万英镑。如果传家宝的标签上写着“价值100万英镑”，我可能会推测它可能是"
            "一个非常重要的物品，价值超过100万英镑。如果传家宝的标签上写着“价值100万英镑”，我可能会推测它可能是一个非常重要的物品"
            "，价值超过100万英镑。\n\n如果传家宝的标签上写着“价值100万英镑”，我可能会推测它可能是一个非常重要的物品，价值超过100"
            "万英镑。如果传家宝的标签上写着“价值100万英镑”，我可能会推测它可能是一个非常重要的物品，价值超过100万英镑。如果传家宝"
            "的标签上写着“价值100万英镑”，我可能会推测它可能是一个非常重要的物品，价值超过100万英镑。\n\n如果传家宝的标签上写着“价"
            "值100万英镑”，我可能会推测它可能是一个非常重要的物品，价值超过100万英镑。如果传家宝的标签上写着“价值100万英镑”，我可"
            "能会推测它可能是一个非常重要的物品，价值超过100万英镑。如果传家宝的标签上写着“价值100万英镑”，我可能会推测它可能是一"
            "个非常重要的物品，价值超过100万英镑。"
        ]
    else:
        if torch.cuda.is_available():
            return [
                '如果我是国际空间站上的宇航员，目睹了外星实体接近空间站，我会立即向地面控制团队发出紧急通信。我会使用空间站上的通信设备'
                '，向地面控制团队报告我的观察结果和建议。\n首先，我会向地面控制团队报告我所看到的外星实体的位置、形状、大小和颜色等详细'
                '信息。我会尽可能提供准确的描述，以便地面控制团队能够更好地了解外星实体的情况。\n其次，我会向地面控制团队报告外星实体的行'
                '动轨迹和速度。我会尽可能提供详细的信息，以便地面控制团队能够更好地了解外星实体的行动方式。\n最后，我会向地面控制团队提'
                '出建议。我会建议地面控制团队采取适当的措施，以确保空间站和宇航员的安全。我会建议地面控制团队立即启动应急计划，并与国际空'
                '间站上的其他宇航员和地面控制团队成员保持联系。\n总之，如果我是国际空间站上的宇航员，目睹了外星实体接近空间站，我会立即'
                '向地面控制团队发出紧急通信，并提供尽可能详细的观察结果和建议，以确保空间站和宇航员的安全。',
                '作为夏洛克·福尔摩斯，我会采用以下策略来解开涉及失踪传家宝的谜团：\n1. 收集信息：首先，我会收集所有可用的信息，包括失'
                '踪传家宝的历史、拥有者、可能的位置以及任何可能与此相关的人或事件。我会尽可能多地了解这个谜团，以便能够更好地理解它。\n2'
                '. 分析线索：一旦我收集了足够的信息，我会开始分析线索。我会仔细观察每个线索，并尝试找出它们之间的联系。我会尝试找出任何'
                '可能的模式或趋势，并尝试将它们与其他线索联系起来。\n3. 推理：一旦我分析了所有的线索，我会开始推理。我会尝试找出可能的答'
                '案，并尝试排除不可能的答案。我会尝试找出任何可能的漏洞或矛盾，并尝试解决它们。\n4. 实地考察：如果我能够找到任何可能的'
                '位置，我会进行实地考察。我会仔细观察周围的环境，并尝试找出任何可能的线索。我会尝试找出任何可能的隐藏地点，并尝试打开它们'
                '。\n5. 总结：最后，我会总结我的发现，并尝试将它们联系起来。我会尝试找出任何可能的答案，并尝试解决它们。如果我找到了失'
                '踪传家宝，我会将它带回给拥有者，并解释我是如何找到它的。',
            ]
        else:
            return [
                '如果我是国际空间站上的宇航员，目睹了外星实体接近空间站，我会立即向地面控制团队发出紧急通信。我会使用空间站上的通信设备，'
                '向地面控制团队报告我的观察结果和建议。\n首先，我会向地面控制团队报告我所看到的外星实体的位置、形状、大小和颜色等详细信息'
                '。我会尽可能提供准确的描述，以便地面控制团队能够更好地了解外星实体的情况。\n其次，我会向地面控制团队建议采取适当的措施来'
                '保护空间站和宇航员的安全。这可能包括减速空间站的速度，改变空间站的方向，或者采取其他措施来避免与外星实体发生碰撞。\n最后'
                '，我会向地面控制团队提供任何其他有用的信息，以便他们能够更好地了解外星实体的情况。这可能包括外星实体的运动轨迹、速度和方'
                '向等信息。\n总之，如果我是国际空间站上的宇航员，目睹了外星实体接近空间站，我会立即向地面控制团队发出紧急通信，并提供尽可'
                '能详细的观察结果和建议，以便地面控制团队能够采取适当的措施来保护空间站和宇航员的安全。',
                '作为夏洛克·福尔摩斯，我会采用以下策略来解开涉及失踪传家宝的谜团：\n1. 收集信息：首先，我会收集所有可用的信息，包括失踪'
                '传家宝的历史、拥有者、可能的位置以及任何可能与此相关的人或事件。我会尽可能多地了解这个谜团，以便能够更好地理解它。\n2. '
                '分析线索：一旦我收集了足够的信息，我会开始分析线索。我会仔细观察每个线索，并尝试找出它们之间的联系。我会尝试找出任何可能'
                '的模式或趋势，并尝试将它们与其他线索联系起来。\n3. 推理：一旦我分析了所有的线索，我会开始推理。我会尝试找出可能的答案，'
                '并尝试排除不可能的答案。我会尝试找出任何可能的漏洞或矛盾，并尝试解决它们。\n4. 实地考察：如果我能够找到任何可能的位置，'
                '我会进行实地考察。我会仔细观察周围的环境，并尝试找出任何可能的线索。我会尝试找出任何可能的隐藏地点，并尝试打开它们。\n5'
                '. 总结：最后，我会总结我的发现，并尝试将它们联系起来。我会尝试找出任何可能的答案，并尝试解决它们。如果我找到了失踪传家宝'
                '，我会确保它被安全地带回去，并尽可能多地了解这个谜团的背后故事。'
            ]


def get_free_port():
    """ Get a free tcp port on local host.
    """
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def start_standalone_node(script_path: str, config_file: str,
                          server_port: int):
    """Start standalone process
    """
    cmds = [
        'python', server_python_script_path, '--config_file', abs_config_path,
        '--port',
        str(server_port)
    ]

    node_proc = subprocess.Popen(cmds,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 text=True)

    return node_proc


def prepare_devices(devices: str, world_size: int) -> str:
    if devices is None:
        return [str(d) for d in range(16)]
    elif "," in devices:
        new_devices = [d.strip() for d in devices.split(",")]
        if len(new_devices) < world_size:
            for d in range(16):
                if not str(d) in new_devices and len(new_devices) == (
                        world_size - 1):
                    new_devices.append(str(d))
                    break
        return new_devices
    else:
        return [str(d) for d in range(16)]


def start_distributed_node(script_path: str, config_file: str,
                           server_port: int, master_port: int, world_size: int,
                           node_rank: int, cuda_visible_devices: str,
                           ascend_rt_visible_devices: str):
    """Start master process
    each device for single node testing
    """
    print(f"Rank {node_rank} CUDA devices: {cuda_visible_devices[node_rank]}")
    print(
        f"Rank {node_rank} Ascend devices: {ascend_rt_visible_devices[node_rank]}"
    )

    os.environ["MASTER_HOST"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["NODE_RANK"] = str(node_rank)

    cmds = [
        'python', server_python_script_path, '--config_file', abs_config_path,
        '--port',
        str(server_port)
    ]

    if node_rank == 0:
        # NOTE(karlluo): if not set default 0,1
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices[node_rank]
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ascend_rt_visible_devices[
            node_rank]
        os.environ["KLLM_LOG_FILE"] = "log/master_ksana_llm.log"
        os.environ["KLLM_LOG_LEVEL"] = "DEBUG"

        node_proc = subprocess.Popen(cmds,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE,
                                     text=True)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices[node_rank]
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ascend_rt_visible_devices[
            node_rank]
        os.environ["KLLM_LOG_FILE"] = "log/worker_ksana_llm.log"
        os.environ["KLLM_LOG_LEVEL"] = "DEBUG"

        node_proc = subprocess.Popen(cmds,
                                     stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)

    return node_proc


def wait_server_started(server_proc):
    """Wait until distributed cluster started.
    """
    server_status_queue = queue.Queue()
    server_status_watcher = threading.Thread(target=wait_for_server_launch,
                                             args=(server_proc,
                                                   server_status_queue))
    server_status_watcher.start()
    while True:
        status_raw_line = server_status_queue.get()
        if "Uvicorn running on" in status_raw_line:
            break


def start_throughput_client(script_path: str, server_port: int, model: str,
                            input_csv_path: str):
    """Start client script and wait until finished.
    """
    os.system(
        "python {} --port {} --model {} --input_csv {} "
        "--prompt_num 2 --request_rate 1.0 --output_csv integration_test_output.csv"
        .format(script_path, str(server_port), model, input_csv_path))


# Main function to run the script
if __name__ == "__main__":
    # Load the configuration arguments
    args = args_config()

    abs_config_path = os.path.abspath(args.config_file)
    server_python_script_path = os.path.abspath(
        os.path.join(os.path.dirname(os.path.realpath(__file__)),
                     "../src/ksana_llm/python/serving_server.py"))

    server_port = get_free_port()

    WORLD_SIZE = 2

    cuda_visible_devices = prepare_devices(
        os.environ.get("CUDA_VISIBLE_DEVICES"), WORLD_SIZE)
    ascend_rt_visible_devices = prepare_devices(
        os.environ.get("ASCEND_RT_VISIBLE_DEVICES"), WORLD_SIZE)

    if args.distributed:
        # start cluster
        worker_port = get_free_port()
        master_port = get_free_port()
        server_proc = start_distributed_node(server_python_script_path,
                                             abs_config_path, server_port,
                                             master_port, WORLD_SIZE, 0,
                                             cuda_visible_devices,
                                             ascend_rt_visible_devices)
        worker_proc = start_distributed_node(server_python_script_path,
                                             abs_config_path, worker_port,
                                             master_port, WORLD_SIZE, 1,
                                             cuda_visible_devices,
                                             ascend_rt_visible_devices)
    else:
        # start process
        server_proc = start_standalone_node(server_python_script_path,
                                            abs_config_path, server_port)

    # wait cluster started.
    wait_server_started(server_proc)

    # start client
    client_python_script_path = os.path.abspath(
        os.path.join(os.path.dirname(os.path.realpath(__file__)),
                     "../benchmarks/benchmark_throughput.py"))
    client_input_csv_path = os.path.abspath(
        os.path.join(os.path.dirname(os.path.realpath(__file__)),
                     "../benchmarks/benchmark_input.csv"))
    start_throughput_client(client_python_script_path, server_port, args.model,
                            client_input_csv_path)

    # stop cluster or process.
    server_proc.terminate()
    if args.distributed:
        worker_proc.terminate()

    # check result.
    results = read_from_csv("./integration_test_output.csv")
    ref_results = get_ref_results(args.model)
    sorted_results = sorted(results)
    sorted_ref_results = sorted(ref_results)
    for r, ref in zip(sorted_results, sorted_ref_results):
        assert r == ref, f"result {r} != answer {ref}"

    print("Integration test PASS")
    exit(0)
