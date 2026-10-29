# TileLang 官方：RMSNorm 整行常驻与分段重读

用两个短实现比较：整行放入 fragment，或在同一 CTA 内分段累计后逆序重读。

## 什么时候选择

- **适用**：FP32、无 learned weight 的 RMS 归一化，epsilon 固定 1e-12；研究宽行的片上资源取舍。
- **不适用**：直接替代 BF16/带权重模型 RMSNorm；把函数名 splitk 理解为跨 CTA 切分。
- **性能依据**：以下收益是源码机制分析，不代表已测得加速；没有本地 GPU 结果。

## 为什么这样设计

| 原来的问题 | 设计选择 | 收益与代价 |
| --- | --- | --- |
| 整行 fragment 占寄存器 | rms_norm 保留整行并一次归一化 | 减少重读；宽行可能溢出 |
| 宽行资源压力大 | rms_norm_splitk 只保存 blk_k 大小的分段 | 减少片上存储；输入读两遍 |
| 第二次读取可能失去缓存 | 逆序处理最近读取的块 | 提高缓存命中的机会；收益需实测 |

**下一步验证**：扫描 N 与 blk_k，检查 register spill、L2 命中及非整 tile；性能与带权重生产 RMSNorm 分开比较。

## 核心完整代码

入口：`rms_norm`, `rms_norm_splitk`。完整两个工厂；同文件默认测试主要运行整行版本，分段版本需要独立覆盖。

```python
import tilelang

import tilelang.language as T

@tilelang.jit(pass_configs={"tl.disable_tma_lower": True})
def rms_norm_splitk(A, blk_m, blk_k):
    M, N = T.const("M, N")
    dtype = T.float

    A: T.Tensor((M, N), dtype)
    B = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(M, blk_m), threads=128) as bx:
        A_shared = T.alloc_shared((blk_m, blk_k), dtype)
        A_local = T.alloc_fragment((blk_m, blk_k), dtype)
        A_powsum = T.alloc_fragment((blk_m,), dtype)

        num_k_step = T.ceildiv(N, blk_k)
        T.clear(A_local)
        for k in T.Serial(num_k_step):
            T.copy(A[bx * blk_m, k * blk_k], A_shared)
            for i, j in T.Parallel(blk_m, blk_k):
                A_local[i, j] += A_shared[i, j] * A_shared[i, j]
        T.reduce_sum(A_local, A_powsum, dim=1)
        for i in T.Parallel(blk_m):
            A_powsum[i] = T.rsqrt(A_powsum[i] / N + 1e-12)

        for k in T.Serial(num_k_step):
            # reverse, better cache hit rate
            T.copy(A[bx * blk_m, (num_k_step - 1 - k) * blk_k], A_shared)
            for i, j in T.Parallel(blk_m, blk_k):
                A_shared[i, j] *= A_powsum[i]
            T.copy(A_shared, B[bx * blk_m, (num_k_step - 1 - k) * blk_k])

    return B

@tilelang.jit(pass_configs={"tl.disable_tma_lower": True})
def rms_norm(A, blk_m):
    M, N = T.const("M, N")
    dtype = T.float

    A: T.Tensor((M, N), dtype)
    B = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(M, blk_m), threads=128) as bx:
        A_local = T.alloc_fragment((blk_m, N), dtype)
        A_pow_local = T.alloc_fragment((blk_m, N), dtype)
        A_powsum = T.alloc_fragment((blk_m,), dtype)

        T.copy(A[bx * blk_m : (bx + 1) * blk_m, :], A_local)
        for i, j in T.Parallel(blk_m, N):
            A_pow_local[i, j] = A_local[i, j] * A_local[i, j]
        T.reduce_sum(A_pow_local, A_powsum, dim=1)
        for i in T.Parallel(blk_m):
            A_powsum[i] = T.rsqrt(A_powsum[i] / N + 1e-12)
        for i, j in T.Parallel(blk_m, N):
            A_local[i, j] *= A_powsum[i]
        T.copy(A_local, B[bx * blk_m : (bx + 1) * blk_m, :])

    return B
```

## 来源与验证

- 来源：[tile-ai/tilelang · rms_norm.py](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/examples/norm/rms_norm.py)，commit `0a8c6716513bcdf8a728735c531c210c33077784`。
- 摘录范围：L2–L2, L3–L3, L6–L36, L39–L62。仅选取核心及其模块内依赖，保留函数体；去除与所选入口无关的外围代码。本文是摘录与分析，非上游原文件。
- 验证：[同文件的参考/检查或评测代码](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/examples/norm/rms_norm.py)；示例测量脚本不是本仓库实测结果，本文未执行。
- 代码版权与许可遵循[上游 LICENSE](../_licenses/tilelang.txt)；代码中的原有版权声明保留。核查日期：2026-10-29。
