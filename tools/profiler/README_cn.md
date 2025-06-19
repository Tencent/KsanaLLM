# 调度profile
可以通过环境变量KLLM_SCHED_TRACE_LEVEL开启schedule event tracer。生成内容以csv格式保存在sched_event_node_xx.csv中，其中xx是NODE_RANK。如果KLLM_SCHED_TRACE_LEVEL=1则记录batch级别的调度结果。KLLM_SCHED_TRACE_LEVEL=2则记录请求级别的调度结果，请求多的时候，生成的文件会比较大。

通过汇集多个csv文件，使用tools/sched_event_trace_process.py来将多个节点的csv文件处理成一个chrome trace格式的trace_events.json文件。

默认以batch为第一维度来输出结果
```
python3 tools/profiler/sched_event_trace_process.py sched_events_node_*.csv -a 
```
也可以通过--node-first设置以node为第一维度来输出结果
```
python3 tools/profiler/sched_event_trace_process.py sched_events_node_*.csv -a --node-first
```
