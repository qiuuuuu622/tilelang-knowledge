# 归约轴、初值与累加精度

> 关键词：reduce_sum / reduce_max / reduce_min / clear / dtype / NaN。
> 适用：示例为 CUDA FP32 行归约；特殊值行为按 API 和目标核查；基线 `0a8c6716513b`，见[版本与来源](../SOURCES.md)。
> 核验：已核查所列上游资料；本文代码仅做 Python 语法检查，未做 TileLang 编译或 GPU 实测。

**归约需要同时声明轴、输出形状、初值和累加精度。**

## 语义与约束

在此快照中，归约输出形状可为去掉归约轴或该轴保留为 1 的形状。输出 buffer 的 dtype 影响累加精度；不能只看输入 dtype。`clear=True` 为当前归约初始化输出；跨块持续累加时应先正确初始化，再明确后续是否保留累加值。

浮点归约会改变加法顺序，精度判断应依据任务定义。max/min 对 NaN 的处理要检查具体后端选项，不能由普通有限数测试推断。指数近似、cast、低精度打包也不是同一个误差来源。

## 最小完整示例

保存为 `row_sum.py`，每个 block 处理一行，没有尾列 padding。

```python
import torch
import tilelang
import tilelang.language as T

@tilelang.jit(target="cuda")
def make_row_sum(M: int = 8, N: int = 256):
    @T.prim_func
    def row_sum(X: T.Tensor((M, N), "float32"),
                Y: T.Tensor((M,), "float32")):
        with T.Kernel(M, threads=128) as row:
            x = T.alloc_fragment((1, N), "float32")
            y = T.alloc_fragment((1,), "float32")
            T.copy(X[row:row + 1, 0:N], x)
            T.reduce_sum(x, y, dim=1, clear=True)
            T.copy(y, Y[row:row + 1])
    return row_sum

if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(8, 256, device="cuda", dtype=torch.float32)
    y = torch.empty(8, device="cuda", dtype=torch.float32)
    make_row_sum()(x, y)
    torch.testing.assert_close(y, x.double().sum(dim=1).float(),
                               rtol=1e-5, atol=1e-4)
    print("PASS")
```

## 为什么这样写

使用二维 fragment 让归约轴可见；以较高精度参考区分普通归约误差和实现错误。扩展为 softmax/RMSNorm 时，需要重新确定数值稳定算法和中间 dtype，不能只串接几个 API。

## 常见误用与检查

| 症状 | 检查动作 |
| --- | --- |
| 输出形状报错 | 写出删轴或 keepdim 后的形状，检查 dim |
| 多轮累加只保留最后一轮 | 检查 `clear=True` 是否放在每轮归约中 |
| 大幅值或抵消输入误差明显 | 对照 FP64 参考，逐步核查输入转换、累加、输出转换 |
| 尾部 max 偏大 | 检查无效列是否被错误填 0 |

## 来源与关联

- [tilelang/language/reduce_op.py](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/tilelang/language/reduce_op.py)
- [testing/python/language/test_tilelang_language_reduce.py](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/testing/python/language/test_tilelang_language_reduce.py)
- [内存与尾块](../semantics/memory-and-boundaries.md) · [RMSNorm 案例](../../kernels/normalization/tilelang-rmsnorm-resident-vs-chunked.md)
