# 怎样得到可比较的 kernel 时间

> 关键词：warmup / CUDA Event / synchronization / latency / benchmark scope。
> 适用：计时代码为 CUDA、单设备、单 stream；不含编译与输入分配；基线 `0a8c6716513b`，见[版本与来源](../SOURCES.md)。
> 核验：已核查所列上游资料；本文代码仅做 Python 语法检查，未做 TileLang 编译或 GPU 实测。

**先确认正确，再明确计时边界，保留多轮原始样本。**

## 最小完整示例

先保存 [vector_add.py](../api/kernel-and-buffers.md)，下面保存为同目录 `bench_add.py`。固定输入和预分配输出，编译、参考计算、正确性检查放在计时之外。

```python
import json
import statistics
import torch
from vector_add import make_add

torch.manual_seed(0)
n = 1 << 20
a = torch.randn(n, device="cuda", dtype=torch.float32)
b = torch.randn_like(a)
c = torch.empty_like(a)
kernel = make_add(n)
kernel(a, b, c)
torch.testing.assert_close(c, a + b, rtol=1e-6, atol=1e-6)
for _ in range(20):
    kernel(a, b, c)
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
end.record()
end.synchronize()
samples_us = []
repeats = 100
for _ in range(7):
    start.record()
    for _ in range(repeats):
        kernel(a, b, c)
    end.record()
    end.synchronize()
    samples_us.append(start.elapsed_time(end) * 1000 / repeats)
print(json.dumps({"samples_us": samples_us,
                  "median_us": statistics.median(samples_us),
                  "min_us": min(samples_us), "max_us": max(samples_us)}))
```

预期输出时间样本和统计量，没有预设目标延迟。该示例测量 stream 上重复调用的平均间隔；短 kernel 可能包含 host 提交导致的空隙，不等同于纯指令执行时间。重复同一输入也会改变缓存状态。

## 为什么这样测

CUDA 操作异步提交，单纯在 Python 调用前后取墙钟可能只量到提交。Event 与执行放在同一 stream，结束事件同步后再取时间。编译和检查不在计时区间，但若真实目标包含输出分配、调度或 graph replay，应另设对应口径，不能混比。

## 比较与记录

| 要固定 | 要记录 |
| --- | --- |
| 输入 shape/dtype/stride、数据、设备与执行模式 | kernel/TileLang commit、配置、硬件与工具链 |
| 是否预分配输出、是否 graph、缓存冷热方式 | warmup、repeats、计时单位、全部样本 |
| 基线与候选的数值要求 | 正确性结果、绝对延迟及相对变化 |

A/B 交替测量并增加独立轮次；若变化与波动同量级，结论写“尚不能区分”。不要从一次最小值宣布提速。profiling、打印与 trace 均关闭后复测；全模型收益还需要独立端到端评测。

## 来源与关联

- [tilelang/profiler/__init__.py](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/tilelang/profiler/__init__.py)
- [docs/programming_guides/autotuning.md](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/docs/programming_guides/autotuning.md)
- [缓存与特化](../compilation/specialization-and-cache.md) · [profiling](profiling.md)
