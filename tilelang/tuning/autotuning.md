# 怎样让 autotune 搜索合法、正确且可复现的候选

> 关键词：autotune / configs / ref_prog / out_idx / set_autotune_inputs / on_trial。
> 适用：示例为 CUDA FP32 向量加；只搜索 block 大小；基线 `0a8c6716513b`，见[版本与来源](../SOURCES.md)。
> 核验：已核查所列上游资料；本文代码仅做 Python 语法检查，未做 TileLang 编译或 GPU 实测。

**搜索器负责尝试配置，使用者负责数学契约、验收输入和结果证据。**

## 语义与约束

- `@tilelang.autotune` 放在 `@tilelang.jit` 上方；configs 的键应对应可调参数。
- 以下例子显式启用 `out_idx=[-1]`：C 由调用包装器分配并返回，因此调用是 `kernel(a,b)`，与无 out_idx 的三参数入门示例不同。
- 默认 profiler 比较路径调用 `ref_prog(*inputs)`，期望返回参考 Tensor/结果序列。不要照搬返回 bool 的检查函数；需要手工比较时再使用相应 manual-check 接口。
- 源码中只有未跳过检查且存在 ref_prog 才执行参考验证；`skip_check=False` 单独出现不是已验证的证据。

## 最小完整示例

保存为 `tune_add.py`；本例仅演示接口闭环，不保证默认测量方式适合所有微小 kernel。

```python
import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs

def reference(a, b):
    return a + b

@tilelang.autotune(
    configs=[{"block": b} for b in (128, 256, 512)],
    ref_prog=reference, rtol=1e-6, atol=1e-6,
    max_mismatched_ratio=0.0, skip_check=False,
)
@tilelang.jit(out_idx=[-1], target="cuda")
def make_add(N: int, block: int = 256):
    @T.prim_func
    def add(A: T.Tensor((N,), "float32"),
            B: T.Tensor((N,), "float32"),
            C: T.Tensor((N,), "float32")):
        with T.Kernel(T.ceildiv(N, block), threads=128) as bx:
            for i in T.Parallel(block):
                gi = bx * block + i
                if gi < N:
                    C[gi] = A[gi] + B[gi]
    return add

if __name__ == "__main__":
    torch.manual_seed(0)
    n = (1 << 20) + 7
    a = torch.randn(n, device="cuda", dtype=torch.float32)
    b = torch.randn_like(a)
    with set_autotune_inputs(a, b):
        kernel = make_add(n)
    c = kernel(a, b)
    torch.testing.assert_close(c, reference(a, b), rtol=1e-6, atol=1e-6)
    print("PASS: selected kernel")
```

## 为什么这样写

固定输入保证候选可比较；显式参考避免“成功运行”被误当作“正确”。本例不修改输入；有原地更新或原子累加的候选需按契约恢复每次运行的初始状态，固定 Tensor 对象不代表内容不变。只搜索一个因素，便于解释结果。GEMM 的 tile/stages/threads 应先筛掉确定非法的配置，再逐步扩展搜索；dtype 或近似算法变化还需重新定义精度约束。

## 结果怎样保存

每轮保存 config、编译/运行状态、检查状态、时间、错误与环境。程序式 `AutoTuner.run(on_trial=...)` 在此快照提供实验性回调，但装饰器没有同样参数；缓存命中只发 marker，不重建历史，也不自动持久化所有 trial。

最终最优项还要经过独立的边界输入回归与 [测量复核](measurement.md)。本例返回 kernel 而未自动导出完整 trial 档案；正式优化任务应使用程序式接口记录回调，或保留逐项实验记录。不要只上传一个 best latency。

## 常见误用与检查

| 误用 | 修正 |
| --- | --- |
| 未提供参考，却宣称 autotune 已验证正确 | 配置 ref_prog 并确认 validation 实际运行 |
| 把 bool 参考直接交给默认比较器 | 返回参考 Tensor；按源码接口提供手工检查 |
| 缓存命中当作新实验 | 记录 cache hit，必要时隔离缓存后重测 |
| 一个 shape 获胜就全局替换 | 回归目标工作负载范围，必要时显式分派 |
| timeout 后马上继续测 | 底层调用未必被强制终止；检查设备状态并恢复干净实验环境 |

## 来源与关联

- [docs/programming_guides/autotuning.md](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/docs/programming_guides/autotuning.md)
- [tilelang/autotuner/tuner.py](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/tilelang/autotuner/tuner.py)
- [tilelang/profiler/__init__.py](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/tilelang/profiler/__init__.py)
- [特化缓存](../compilation/specialization-and-cache.md) · [工作负载特化模式](../../patterns/parallelism/specialize-by-workload.md)
