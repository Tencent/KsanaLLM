# 如何启动 Expert-Parallel EP 并行程序

## 1. 环境搭建

### 1.1 容器创建

```
sudo docker run -itd \
  --name yourname_deepep \
  -v /data1/:/data1 \
  --privileged \
  --gpus all \
  -dit \
  --net=host \
  --uts=host \
  --ipc=host \
  --security-opt=seccomp=unconfined \
  --device=/dev/infiniband \
  -v /sys/class:/sys/class \
  -v /lib/modules:/lib/modules \
  -v /usr/src:/usr/src \
  -v /sys/devices:/sys/devices \
  -v /etc/machine-id:/etc/machine-id:ro \
  mirrors.tencent.com/todacc/venus-std-base-tlinux4-ksana-hopper-gpu-deepep:0.1.4

sudo docker exec -it yourname_deepep /bin/bash
ln -sf /data1/models /model
pip install triton==3.2.0
```

### 1.2 KsanaLLM 构建

```
cd /work
git clone https://git.woa.com/RondaServing/LLM/KsanaLLM.git
mkdir -p KsanaLLM/build
cd KsanaLLM/build
cmake -DSM=90a -DWITH_VLLM_FLASH_ATTN=ON .. && make -j
```

## 2. 服务启动

### 2.1 DeepEPWrapper 启动

- 需要按照需求，传入需要的配置，脚本入参说明:
  - ``CUDA_VISIBLE_DEVICES=xx ./bin/DeepEPWrapper A B C``
  - 建议在每次启动时，手动声明 CUDA_VISIBLE_DEVICES，明确 deepseek_wrapper 启动用卡
  - DeepEPWrapper 的三个输入项，A/B/C 均为 int 类型正整数，分别代表如下含义
    - A: 总共有多少张卡参与 EP 并行
    - B: world_size，总机器个数
    - C: node_rank，当前机器序号
  - 单机双卡示例：
    - ``CUDA_VISIBLE_DEVICES=0,1 ./bin/DeepEPWrapper 2``
  - 双机16卡示例：
    - ```shell
      # 节点 0
      CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./bin/DeepEPWrapper 16 2 0

      # 节点 1
      CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./bin/DeepEPWrapper 16 2 1
      ```
  - 启动后，服务将监听等待一念进程
    - ```
        === Running DeepEPWrapper ===
        Args:  4 2 0
        CUDA devices: 0, 1
        =============================
        ===== deepep_wqrapper =====
        waiting for open  shared memory
      ```

### 2.2 一念启动

- 当前版本中，需要先启动 deepseek_wrapper ，确保其处于监听状态后，再启动一念进程(TBD)
- ```
  python ../src/ksana_llm/python/serving_server.py \
    --config_file ksana_llm.yaml \
    --port 6543
  ```

```

## 3. 服务终止
- 使用辅助脚本， 清空所有进程（TBD:一念与deepep_wrappers生命周期联动）
- ```
  python kill_server.py Multi
python kill_server.py sh run.sh
python kill_server.py python
rm -f *_log.txt
```

## 3.1 辅助脚本 kill_server.py

```
import sys
import subprocess
import psutil

def kill_other_execute(kill_type=["default pid"]):
    # 获取进程列表
    try:
        # 执行ps命令获取进程信息
        ps_output = subprocess.check_output(['ps', '-ef'], universal_newlines=True)

        # 筛选匹配条件的进程
        pids = []
        for line in ps_output.splitlines():
            need_kill = True
            for kill_target in kill_type:
                if kill_target not in line:
                    need_kill = False
                    break
                if "root           1       0" in line:
                    need_kill = False
                    break
                if "ceph-fuse " in line:
                    need_kill = False
                    break
                if "python" in line and "kill_server" in line:
                    need_kill = False
                    break
            if need_kill:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        pid = int(parts[1])
                        pids.append(pid)
                    except ValueError:
                        continue
        # 杀死筛选出的进程
        for pid in pids:
            try:
                parent = psutil.Process(pid)
                for child in parent.children(recursive=True):
                    try:
                        child.kill()
                        print(f"已终止子进程 {child.pid}")
                    except psutil.NoSuchProcess:
                        print(f"子进程 {child.pid} 不存在")
                    except psutil.AccessDenied:
                        print(f"没有权限终止子进程 {child.pid}")

                # 再杀死父进程
                parent.kill()
                parent.wait()
                print(f"已终止父进程 {pid}")
            except psutil.NoSuchProcess:
                print(f"进程 {pid} 不存在")
            except psutil.AccessDenied:
                print(f"没有权限终止进程 {pid}")
    except subprocess.SubprocessError as e:
        print(f"获取进程列表时出错: {e}")

pid = sys.argv[1:]
if len(pid) > 0:
    kill_other_execute(pid)
```
