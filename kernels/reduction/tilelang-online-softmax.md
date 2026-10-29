# TileLang 官方：长行 online softmax

第一遍按块更新 log-sum-exp，第二遍重读输入并输出归一化结果，限制片上状态大小。

## 什么时候选择

- **适用**：FP16、二维行归一化；原例 M=N=8192，先保持 N 与 BLOCK_N 整除再扩展。
- **不适用**：未经处理的非整 tile：普通零填充不能作为 softmax 的无效列值；必须核查 mask。
- **性能依据**：以下收益是源码机制分析，不代表已测得加速；没有本地 GPU 结果。

## 为什么这样设计

| 原来的问题 | 设计选择 | 收益与代价 |
| --- | --- | --- |
| 整行无法经济地常驻 | 逐块合并 LSE | 状态与块宽相关；第一遍仍需扫描全行 |
| 稳定性要求避免大指数 | 块最大值与 log2/exp2 递推 | 控制指数范围；浮点误差需对照 |
| 最终分母在末尾才知道 | 第二遍读取并写输出 | 无需完整中间张量；多读一次输入 |

**下一步验证**：先测试全负值和尾部 mask，避免零 padding 被误计入分母；再比较短/长行下两遍访存与寄存器节省。

## 核心完整代码

入口：`softmax_kernel`。保留完整函数及原始 M/N 常量；不擅自把固定形状示例改写成动态通用 API。

```python
import tilelang

import tilelang.language as T

@tilelang.jit
def softmax_kernel(X, BLOCK_M=1, BLOCK_N=8192, dtype: T.dtype = T.float16):
    X: T.Tensor([M, N], dtype)
    Y = T.empty([M, N], dtype)

    accum_dtype = T.float32

    scale = 1.44269504  # log2(e)

    with T.Kernel(T.ceildiv(M, BLOCK_M), threads=128) as (i_m):
        x = T.alloc_fragment([BLOCK_M, BLOCK_N], dtype)
        y = T.alloc_fragment([BLOCK_M, BLOCK_N], dtype)
        lse = T.alloc_fragment([BLOCK_M], accum_dtype)
        max_x = T.alloc_fragment([BLOCK_M], dtype)
        exp_x = T.alloc_fragment([BLOCK_M, BLOCK_N], accum_dtype)
        sum_exp_x = T.alloc_fragment([BLOCK_M], accum_dtype)
        T.fill(lse, -T.infinity(accum_dtype))

        for i_n in T.Pipelined(T.ceildiv(N, BLOCK_N)):
            T.copy(X[i_m * BLOCK_M, i_n * BLOCK_N], x)
            T.reduce_max(x, max_x, dim=1, clear=True)

            for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                exp_x[i, j] = T.exp2(x[i, j] * scale - max_x[i] * scale)

            T.reduce_sum(exp_x, sum_exp_x, dim=1, clear=True)

            for i in T.Parallel(BLOCK_M):
                lse[i] = max_x[i] * scale + T.log2(T.exp2(lse[i] - max_x[i] * scale) + sum_exp_x[i])

        for i_n in T.Pipelined(T.ceildiv(N, BLOCK_N)):
            T.copy(X[i_m * BLOCK_M, i_n * BLOCK_N], x)

            for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                y[i, j] = T.exp2(x[i, j] * scale - lse[i])

            T.copy(y, Y[i_m * BLOCK_M, i_n * BLOCK_N])

    return Y

M = 8192

N = 8192
```

## 来源与验证

- 来源：[tile-ai/tilelang · online_softmax.py](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/examples/online_softmax/online_softmax.py)，commit `0a8c6716513bcdf8a728735c531c210c33077784`。
- 摘录范围：L2–L2, L3–L3, L7–L45, L48–L48, L49–L49。仅选取核心及其模块内依赖，保留函数体；去除与所选入口无关的外围代码。本文是摘录与分析，非上游原文件。
- 验证：[同文件的参考/检查或评测代码](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/examples/online_softmax/online_softmax.py)；示例测量脚本不是本仓库实测结果，本文未执行。
- 代码版权与许可遵循[上游 LICENSE](../_licenses/tilelang.txt)；代码中的原有版权声明保留。核查日期：2026-10-29。
