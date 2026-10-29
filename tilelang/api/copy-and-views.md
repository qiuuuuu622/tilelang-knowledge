# 复制区域与 view 的边界

> 关键词：T.copy / BufferRegion / BufferLoad / stride / async_copy。
> 适用：示例为 CUDA FP32 连续向量；不含非整 tile；基线 `0a8c6716513b`，见[版本与来源](../SOURCES.md)。
> 核验：已核查所列上游资料；本文代码仅做 Python 语法检查，未做 TileLang 编译或 GPU 实测。

**先明确源、目标覆盖的元素集合，再讨论搬运指令。view 改变解释方式，copy 才表达数据搬运。**

## 语义与约束

`T.copy` 接受 Buffer、BufferRegion 与可推导区域的 BufferLoad。完整 Buffer 两侧应形状匹配；区域起点写法可借另一侧推导 extent，但不应将内部形状补齐规则当成一般广播能力。源码明确提示不匹配 extent 可能产生意外结果。

非连续输入需明确 stride 契约；创建 view 不会自动把源数据重排成连续布局。dtype 转换、向量宽度和地址对齐也需要检查。

## 最小完整示例

保存为 `tile_copy.py`。这里只演示显式匹配的区域与 shared 生命周期，不宣称比直接复制更快。

```python
import torch
import tilelang
import tilelang.language as T

@tilelang.jit(target="cuda")
def make_copy(N: int, block: int = 256):
    assert N > 0 and N % block == 0
    @T.prim_func
    def copy_kernel(A: T.Tensor((N,), "float32"),
                    B: T.Tensor((N,), "float32")):
        with T.Kernel(N // block, threads=128) as bx:
            shared = T.alloc_shared((block,), "float32")
            T.copy(A[bx * block:(bx + 1) * block], shared)
            T.copy(shared, B[bx * block:(bx + 1) * block])
    return copy_kernel

if __name__ == "__main__":
    a = torch.arange(1024, device="cuda", dtype=torch.float32)
    b = torch.empty_like(a)
    make_copy(1024)(a, b)
    torch.testing.assert_close(b, a, rtol=0, atol=0)
    print("PASS")
```

## 为什么这样写

区域长度两侧一致，便于追踪范围；选择整 tile，避免将复制协议与 padding 数学语义混在第一个例子中。扩展尾块时需要单独验证边界，扩展异步时需要单独设计等待。

## 常见误用与检查

| 误用或症状 | 检查 / 修正 |
| --- | --- |
| 将 `[M,1]` 当成可自动广播到 `[M,N]` | 显式逐元素表达广播，或证明该具体复制区域合法 |
| 把 `T.copy` 替换成 async 就期望提速 | 先读同步契约；立即等待也不保证有重叠收益 |
| 切片输入只检查 shape | 同时记录 stride、storage offset 与目标对齐约束 |
| 根据调用名称判断用了 TMA | 检查生成源码与目标架构，不能仅靠 API 名字 |

## 来源与关联

- [tilelang/language/copy_op.py](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/tilelang/language/copy_op.py)
- [docs/programming_guides/instructions.md](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/docs/programming_guides/instructions.md)
- [同步](../semantics/synchronization.md) · [布局与访存模式](../../patterns/data-movement/match-layout-to-access.md)
