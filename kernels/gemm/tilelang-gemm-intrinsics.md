# TileLang 官方：显式 Tensor Core 指令与布局

显式组织片段布局、装载与 MMA 指令，展示高层 T.gemm 以下的控制点。

## 什么时候选择

- **适用**：已经从生成代码定位到布局或指令问题，愿意绑定 TensorCoreIntrinEmitter 的 tile/warp 契约。
- **不适用**：常规 GEMM 入门替代、任意 K 尾部；源码 K//block_K 循环需保证整除。
- **性能依据**：以下收益是源码机制分析，不代表已测得加速；没有本地 GPU 结果。

## 为什么这样设计

| 原来的问题 | 设计选择 | 收益与代价 |
| --- | --- | --- |
| 默认布局未必匹配目标指令 | make_swizzle_layout 与 emitter 显式组织数据 | 增加控制力；绑定 dtype/指令形状 |
| 跨层难以定位性能来源 | 显式 load、mma、store 调用 | 便于对照生成代码；维护复杂度提高 |
| 寄存器和 warp tile 相互制约 | 固定 warp 分解和局部片段 | 控制复用；参数不能独立随意修改 |

**下一步验证**：保持计算规模相同，与 T.gemm 版本比较 SASS、bank conflict 和寄存器；只在证据支持时保留手工版本。

## 核心完整代码

入口：`tl_matmul`。完整 tl_matmul 及布局辅助函数；使用固定版本 TileLang 的 intrinsics 接口。

```python
from tvm import DataType

import tilelang

import tilelang.language as T

from tilelang.cuda.intrinsics import get_swizzle_layout

from tilelang.cuda.intrinsics.macro.mma_macro_generator import (
    TensorCoreIntrinEmitter,
)

def make_swizzle_layout(shared_buf):
    dtype = shared_buf.dtype
    shape = shared_buf.shape

    can_swizzle = shape[-1] * DataType(dtype).bits == 512
    if not can_swizzle:
        return T.Layout(shape, lambda *args: args)

    def transform_func(i, j):
        return list(get_swizzle_layout(i, j, shape[-1], dtype))

    return T.Layout(shape, transform_func)

@tilelang.jit
def tl_matmul(
    A,
    B,
    in_dtype,
    out_dtype,
    accum_dtype,
):
    assert in_dtype in [
        T.float16,
        T.int8,
    ], "Currently only float16 and int8 are supported"
    assert out_dtype in [
        T.float16,
        T.float32,
        T.int32,
    ], "Currently only float16, float32 and int32 are supported"

    micro_size_x = micro_size_y = micro_size_k = 16

    if out_dtype == T.int32:
        micro_size_k = 32

    # This is a debug config
    block_row_warps = 2
    block_col_warps = 2
    warp_row_tiles = 64
    warp_col_tiles = 64
    # chunk = 32 if in_dtype == T.float16 else 64
    chunk = 32
    shared_scope = "shared.dyn"

    # Pipeline Stage
    stage = 2

    block_M = block_row_warps * warp_row_tiles
    block_N = block_col_warps * warp_col_tiles
    block_K = chunk

    M, N, K = T.const("M, N, K")
    A_shape = (M, K)
    B_shape = (N, K)
    A_shared_shape = (block_M, block_K)
    B_shared_shape = (block_N, block_K)
    C_shared_shape = (
        block_M // micro_size_x,
        block_N // micro_size_y,
        micro_size_x,
        micro_size_y,
    )

    A: T.Tensor(A_shape, in_dtype)
    B: T.Tensor(B_shape, in_dtype)
    C = T.empty((M, N), out_dtype)

    warp_size = 32
    threads = warp_size * (block_row_warps * block_col_warps)
    local_size_a = (micro_size_x * micro_size_k) // warp_size
    local_size_b = (micro_size_y * micro_size_k) // warp_size
    local_size_c = (micro_size_x * micro_size_y) // warp_size
    warp_rows = warp_row_tiles // micro_size_x
    warp_cols = warp_col_tiles // micro_size_y

    # MMA Wrapper to Auto Generate Code for MMA
    mma_emitter = TensorCoreIntrinEmitter(
        a_dtype=in_dtype,
        b_dtype=in_dtype,
        accum_dtype=accum_dtype,
        a_transposed=False,
        b_transposed=True,
        block_row_warps=block_row_warps,
        block_col_warps=block_col_warps,
        warp_row_tiles=warp_row_tiles,
        warp_col_tiles=warp_col_tiles,
        chunk=chunk,
    )

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
        A_shared = T.alloc_shared(A_shared_shape, in_dtype, scope=shared_scope)
        B_shared = T.alloc_shared(B_shared_shape, in_dtype, scope=shared_scope)
        C_shared = T.alloc_shared(C_shared_shape, out_dtype, scope=shared_scope)
        A_local = T.alloc_local((warp_rows * local_size_a), in_dtype)
        B_local = T.alloc_local((warp_cols * local_size_b), in_dtype)
        C_local = T.alloc_local((warp_rows * warp_cols * local_size_c), accum_dtype)

        T.annotate_layout(
            {
                A_shared: make_swizzle_layout(A_shared),
                B_shared: make_swizzle_layout(B_shared),
            }
        )

        # Improve L2 Cache
        T.use_swizzle(panel_size=10)

        T.clear(C_local)

        for ko in T.Pipelined((K // block_K), num_stages=stage):
            # Load A into shared memory
            for i, k in T.Parallel(block_M, block_K):
                A_shared[i, k] = A[by * block_M + i, ko * block_K + k]

            # Load B into shared memory
            for j, k in T.Parallel(block_N, block_K):
                B_shared[j, k] = B[bx * block_N + j, ko * block_K + k]

            for ki in T.serial(0, (block_K // micro_size_k)):
                # Load A into fragment
                mma_emitter.ldmatrix_a(A_local, A_shared, ki)

                # Load B into fragment
                mma_emitter.ldmatrix_b(B_local, B_shared, ki)

                # Perform Matrix Multiplication
                mma_emitter.mma(A_local, B_local, C_local)

        # Perform STMatrix
        mma_emitter.stmatrix(C_local, C_shared)

        # Store shared into global
        for i, j in T.Parallel(block_M, block_N):
            C[by * block_M + i, bx * block_N + j] = C_shared[
                i // micro_size_x,
                j // micro_size_y,
                i % micro_size_x,
                j % micro_size_y,
            ]

    return C
```

## 来源与验证

- 来源：[tile-ai/tilelang · example_gemm_intrinsics.py](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/examples/gemm/example_gemm_intrinsics.py)，commit `0a8c6716513bcdf8a728735c531c210c33077784`。
- 摘录范围：L2–L2, L3–L3, L4–L4, L5–L5, L6–L8, L11–L22, L25–L153。仅选取核心及其模块内依赖，保留函数体；去除与所选入口无关的外围代码。本文是摘录与分析，非上游原文件。
- 验证：[同文件的参考/检查或评测代码](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/examples/gemm/example_gemm_intrinsics.py)；示例测量脚本不是本仓库实测结果，本文未执行。
- 代码版权与许可遵循[上游 LICENSE](../_licenses/tilelang.txt)；代码中的原有版权声明保留。核查日期：2026-10-29。
