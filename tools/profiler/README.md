# Scheduling Profile
The schedule event tracer can be enabled through the environment variable KLLM_SCHED_TRACE_LEVEL. The generated content is saved in CSV format as sched_event_node_xx.csv, where xx represents the NODE_RANK.

When KLLM_SCHED_TRACE_LEVEL=1, batch-level scheduling results are recorded
When KLLM_SCHED_TRACE_LEVEL=2, request-level scheduling results are recorded (note that when there are many requests, the generated files can become quite large)
By aggregating multiple CSV files, you can use tools/sched_event_trace_process.py to process CSV files from multiple nodes into a single Chrome trace format file named trace_events.json.

By default, results are output with batch as the primary dimension:
```bash
python3 tools/profiler/sched_event_trace_process.py sched_events_node_*.csv -a 
```
Alternatively, you can use the --node-first flag to output results with node as the primary dimension:
```bash
python3 tools/profiler/sched_event_trace_process.py sched_events_node_*.csv -a --node-first
```