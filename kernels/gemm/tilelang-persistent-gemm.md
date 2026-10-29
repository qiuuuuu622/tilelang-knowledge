# TileLang 官方：persistent GEMM 对照

对照每 tile 一个 CTA 与按 SM 数启动 CTA、循环领取 tile 两种调度方式。

## 什么时候选择

- **适用**：大量输出 tile、明确 SM 数和相同 GEMM 精度；需要实验调度顺序与缓存复用时。
- **不适用**：很小的 tile 总数，或把每 SM 一个 CTA 当成普遍最优占用率。
- **性能依据**：以下收益是源码机制分析，不代表已测得加速；没有本地 GPU 结果。

## 为什么这样设计

| 原来的问题 | 设计选择 | 收益与代价 |
| --- | --- | --- |
| tile 多时调度方式影响缓存 | 固定 CTA 数，通过 T.Persistent 遍历 | 改变调度/复用；可能限制并发 CTA 数 |
| 相邻 tile 可复用输入 | 手工分组顺序与 persistent primitive 对照 | 探索局部性；手工分组存在边界约束 |
| CTA 连续处理不同输出 | 每个 tile 前 T.clear(C_local) | 避免累积串台；不能只在 CTA 开头清零 |

**下一步验证**：先检查手工 group_size=8 分支对不整组 N 的覆盖，再比较普通/persistent 的完整输出、L2 命中与时间。

## 核心完整代码

入口：`matmul_non_persistent`, `matmul_persistent`。保留两个完整工厂和原始手工调度分支；先用 primitive 路径，手工分组的非整组覆盖需独立验证。

```python
import tilelang

import tilelang.language as T

from tilelang.carver.arch import driver

@tilelang.jit
def matmul_non_persistent(A, B, block_M, block_N, block_K, threads, num_stages, dtype=T.float16, accum_dtype=T.float32):
    M, N, K = T.const("M, N, K")

    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=threads) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
        C_shared = T.alloc_shared((block_M, block_N), dtype)

        T.use_swizzle(10)

        T.clear(C_local)
        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
            T.copy(A[bx * block_M, k * block_K], A_shared)
            T.copy(B[k * block_K, by * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        T.copy(C_local, C_shared)
        T.copy(C_shared, C[bx * block_M, by * block_N])

    return C

@tilelang.jit
def matmul_persistent(
    A, B, block_M, block_N, block_K, threads, num_stages, dtype=T.float16, accum_dtype=T.float32, use_persistent_primitive=True
):
    M, N, K = T.const("M, N, K")

    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C = T.empty((M, N), dtype)

    sm_num = driver.get_num_sms()
    m_blocks = T.ceildiv(M, block_M)
    n_blocks = T.ceildiv(N, block_N)
    waves = T.ceildiv(m_blocks * n_blocks, sm_num)
    group_size = 8

    with T.Kernel(sm_num, threads=threads) as (block_id):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
        C_shared = T.alloc_shared((block_M, block_N), dtype)

        if use_persistent_primitive:
            for bx, by in T.Persistent([T.ceildiv(M, block_M), T.ceildiv(N, block_N)], sm_num, block_id):
                T.clear(C_local)
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    T.copy(A[bx * block_M, k * block_K], A_shared)
                    T.copy(B[k * block_K, by * block_N], B_shared)
                    T.gemm(A_shared, B_shared, C_local)

                T.copy(C_local, C_shared)
                T.copy(C_shared, C[bx * block_M, by * block_N])
        else:
            for w in T.serial(waves):
                tile_id = sm_num * w + block_id
                bx = (tile_id // group_size) % m_blocks
                by = (tile_id % group_size) + (tile_id // group_size) // m_blocks * group_size

                if bx * block_M < M and by * block_N < N:
                    T.clear(C_local)
                    for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                        T.copy(A[bx * block_M, k * block_K], A_shared)
                        T.copy(B[k * block_K, by * block_N], B_shared)
                        T.gemm(A_shared, B_shared, C_local)

                    T.copy(C_local, C_shared)
                    T.copy(C_shared, C[bx * block_M, by * block_N])

    return C
```

## 来源与验证

- 来源：[tile-ai/tilelang · example_gemm_persistent.py](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/examples/gemm/example_gemm_persistent.py)，commit `0a8c6716513bcdf8a728735c531c210c33077784`。
- 摘录范围：L1–L1, L2–L2, L3–L3, L7–L32, L35–L83。仅选取核心及其模块内依赖，保留函数体；去除与所选入口无关的外围代码。本文是摘录与分析，非上游原文件。
- 验证：[同文件的参考/检查或评测代码](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/examples/gemm/example_gemm_persistent.py)；示例测量脚本不是本仓库实测结果，本文未执行。
- 代码版权与许可遵循[上游 LICENSE](../_licenses/tilelang.txt)；代码中的原有版权声明保留。核查日期：2026-10-29。
