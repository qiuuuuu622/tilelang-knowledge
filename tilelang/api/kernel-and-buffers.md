# 定义、调用并检查第一个 kernel

> 关键词：tilelang.jit / T.prim_func / T.Tensor / T.Kernel / explicit output。
> 适用：示例为 CUDA；固定版本 TileLang、兼容 PyTorch 和 NVIDIA GPU；基线 `0a8c6716513b`，见[版本与来源](../SOURCES.md)。
> 核验：已核查所列上游资料；本文代码仅做 Python 语法检查，未做 TileLang 编译或 GPU 实测。

**先用显式输入输出的 factory 写法分清编译与调用；所有后续示例保持同一约定。**

## 语义与约束

`make_add(N, block)` 返回已特化的可调用 kernel；`kernel(a, b, c)` 将结果写入 c。示例没有 `out_idx`，输出由调用者分配，不依赖自动返回 Tensor。输入是连续的一维 FP32 Tensor；N 为正整数。

## 最小完整示例

保存代码为 `vector_add.py`，在目标 GPU 环境执行 `python vector_add.py`。其他条目引用此文件时，先将此代码保存到同一实验目录。

```python
import torch
import tilelang
import tilelang.language as T

@tilelang.jit(target="cuda")
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
    n = 1031
    a = torch.randn(n, device="cuda", dtype=torch.float32)
    b = torch.randn_like(a)
    c = torch.empty_like(a)
    kernel = make_add(n)
    kernel(a, b, c)
    torch.testing.assert_close(c, a + b, rtol=1e-6, atol=1e-6)
    torch.cuda.synchronize()
    print("PASS", tuple(c.shape))
```

预期 `PASS (1031,)`；这是预期行为，非本仓库执行结果。

## 为什么这样写

显式 guard 使尾块契约可见；主入口保护避免 import 时分配 GPU 数据。这里不引入 shared，因为每个输入只读一次，搬到 shared 没有本例需要的数据复用理由。

现有 `kernels/` 中部分代码使用 eager JIT：直接接收 Tensor、`T.empty` 创建输出并返回。两种形式都存在；迁移时应整体保留一套参数与返回约定，不能只替换装饰器。

## 常见误用与检查

| 误用 | 修正 |
| --- | --- |
| 给 factory 传 Tensor、却期待得到输出 | 区分特化调用与设备调用；按上面两行使用 |
| 忘记传 c | 本例输出是显式参数，并未启用 `out_idx` |
| N、dtype 与实际 Tensor 不符 | 在调用前检查 shape/dtype/device/stride |
| 导入文件时就运行 | 将分配与调用保留在 main guard 内 |

## 来源与关联

- [docs/programming_guides/language_basics.md](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/docs/programming_guides/language_basics.md)
- [tilelang/jit/__init__.py](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/tilelang/jit/__init__.py)
- [执行模型](../semantics/execution-model.md) · [测量](../tuning/measurement.md)
