# TileOPs：softmax 常驻与流式读取

对可常驻的行直接归一化，对超长行以在线 max/sum 两遍扫描，并保留部分头部数据减少重读。

## 什么时候选择

- **适用**：二维连续行的 softmax/log_softmax，dtype 与 factory 参数一致；由调用方提供合法 CTA、线程和缓存配置。
- **不适用**：把 log_softmax 与 softmax 的片上需求当成相同；手工指定过大的 held_bytes 或缺少 SM 约束。
- **性能依据**：以下收益是源码机制分析，不代表已测得加速；没有本地 GPU 结果。

## 为什么这样设计

| 原来的问题 | 设计选择 | 收益与代价 |
| --- | --- | --- |
| softmax 与 log_softmax 对原始 x 的复用不同 | softmax 直接 fragment 输出，log_softmax 暂存原输入 | 按数据生命周期选择存储；两条路径资源不同 |
| 超长行不能完整常驻 | persistent CTA 在线归约再重读 | 控制片上状态；输入需要第二遍 |
| 第二遍容易失去缓存 | 逆序扫描并将部分头块留在 shared | 降低部分重读成本；额外 shared 会影响驻留 |

**下一步验证**：先测非对齐列、全 -inf 与极值，区分库对异常行的约定；再扫描行数、行宽、held_bytes 并观察 spill/L2/时间。

## 核心完整代码

入口：`_softmax_kernel_single`, `_softmax_streaming_kernel`。完整两个 factory 与本文件辅助函数；依赖 TileOPs 的 exp_shifted、streaming_load 和常量。上游还含跨块 split 方案，本案例聚焦常驻/流式这组取舍。

```python
import functools

import tilelang

import tilelang.language as T

from tvm import DataType

from tileops._csrc import csrc_include

from tileops._kernels.constants import VECTOR_ACCESS_BYTES

from tileops._kernels.reduction._primitives import (
    AUTOTUNE_THREADS,
    DEFAULT_ALIGNMENT,
    DEFAULT_THREADS,
    FRAGMENT_ELEMS_PER_THREAD,
    BlockConfigPlanner,
    RowTiledAutotuneMixin,
    ceildiv_int,
    exp_shifted,
    restore_same_shape,
    rows_for_axes,
)

from tileops._kernels.streaming_load import load16_evict_last

from tileops._kernels.tiling import align_up

from tileops._utils import WARP_LANES, WARP_SHUFFLE_STAGES

@functools.lru_cache(maxsize=32)
def _softmax_streaming_kernel(M, N, op_kind, dtype, out_dtype, ctas, ctas_per_sm, held_bytes):
    """Build the two-pass softmax/log_softmax for rows no CTA holds on chip.

    ``ctas`` resident CTAs walk the rows in turn. The first pass reads a row in tiles of
    ``threads * accesses`` vectors and folds each thread's running ``(max, sum)`` with the
    online recurrence; the pairs then fold across the warp by shuffles and across warps
    through shared memory. The second pass reads the row again from the far end, where
    the tiles read last are likeliest still in L2, and writes the result. The first
    tiles of a row, as many whole ones as ``held_bytes`` of shared memory take, are kept
    from the first pass, so the second reads them from shared memory instead.
    """
    vec = VECTOR_ACCESS_BYTES // (DataType(dtype).bits // 8)
    while N % vec:
        vec //= 2
    neg_inf = float("-inf")

    @tilelang.jit(out_idx=[1])
    def _func(threads, accesses):
        tile = threads * accesses * vec
        tiles = -(-N // tile)
        warps = threads // WARP_LANES
        kept = min(N // tile, held_bytes // (tile * DataType(dtype).bits // 8))

        def row_scale(total):
            """What a row's result takes from its sum: its reciprocal, or its log."""
            if op_kind == "softmax":
                return 1.0 / total
            return T.log(total)

        def fold(m, s, other_m, other_s):
            """The ``(max, sum)`` of two partial pairs; two empty ones stay empty."""
            top = T.max(m, other_m)
            both = s * exp_shifted(m, top) + other_s * exp_shifted(other_m, top)
            return top, T.if_then_else(top == neg_inf, T.cast(0, "float32"), both)

        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            y: T.Tensor[(M, N), out_dtype],
        ):
            with T.Kernel(ctas, threads=threads) as cta:
                # Held to the registers that leave room for every CTA an SM is given.
                T.annotate_min_blocks_per_sm(ctas_per_sm)
                tx = T.get_thread_binding()
                held = T.alloc_local([accesses * vec], dtype)
                out = T.alloc_local([vec], out_dtype)
                stat = T.alloc_local([2], "float32")  # this thread's (max, sum)
                peer = T.alloc_local([2], "float32")
                tile_max = T.alloc_local([1], "float32")
                # Two slots, so a row's pairs never overwrite the ones still being read.
                warp_stats = T.alloc_shared([2, 2, warps], "float32")
                head = T.alloc_shared([max(kept, 1) * tile], dtype)

                for step in T.serial(T.ceildiv(M - cta, ctas)):
                    row = cta + step * ctas
                    stat[0] = T.cast(neg_inf, "float32")
                    stat[1] = T.cast(0, "float32")
                    for t in T.serial(tiles):
                        for a in T.unroll(accesses):
                            col = t * tile + (a * threads + tx) * vec
                            if col < N:
                                for i in T.vectorized(vec):
                                    held[a * vec + i] = x[row, col + i]
                            else:
                                for i in T.vectorized(vec):
                                    held[a * vec + i] = T.cast(neg_inf, dtype)
                            if t < kept:
                                for i in T.vectorized(vec):
                                    head[t * tile + (a * threads + tx) * vec + i] = held[
                                        a * vec + i
                                    ]
                        tile_max[0] = T.cast(neg_inf, "float32")
                        for j in T.unroll(accesses * vec):
                            tile_max[0] = T.max(tile_max[0], T.cast(held[j], "float32"))
                        top = T.max(stat[0], tile_max[0])
                        stat[1] = T.if_then_else(
                            top == neg_inf,
                            T.cast(0, "float32"),
                            stat[1] * exp_shifted(stat[0], top),
                        )
                        stat[0] = top
                        for j in T.unroll(accesses * vec):
                            stat[1] += T.if_then_else(
                                top == neg_inf,
                                T.cast(0, "float32"),
                                exp_shifted(T.cast(held[j], "float32"), top),
                            )
                    for k in T.unroll(WARP_LANES.bit_length() - 1):
                        peer[0] = T.shfl_xor(stat[0], T.shift_left(1, k))
                        peer[1] = T.shfl_xor(stat[1], T.shift_left(1, k))
                        stat[0], stat[1] = fold(stat[0], stat[1], peer[0], peer[1])
                    if tx % WARP_LANES == 0:
                        warp_stats[step % 2, 0, tx // WARP_LANES] = stat[0]
                        warp_stats[step % 2, 1, tx // WARP_LANES] = stat[1]
                    T.sync_threads()
                    stat[0] = T.cast(neg_inf, "float32")
                    stat[1] = T.cast(0, "float32")
                    for w in T.unroll(warps):
                        stat[0], stat[1] = fold(
                            stat[0], stat[1], warp_stats[step % 2, 0, w], warp_stats[step % 2, 1, w]
                        )
                    scale = row_scale(stat[1])

                    for t_back in T.serial(tiles):
                        t = tiles - 1 - t_back
                        for a in T.unroll(accesses):
                            col = t * tile + (a * threads + tx) * vec
                            if col < N:
                                if t < kept:
                                    for i in T.vectorized(vec):
                                        held[a * vec + i] = head[col + i]
                                else:
                                    for i in T.vectorized(vec):
                                        held[a * vec + i] = x[row, col + i]
                                for i in T.unroll(vec):
                                    v = T.cast(held[a * vec + i], "float32")
                                    if op_kind == "softmax":
                                        out[i] = T.cast(exp_shifted(v, stat[0]) * scale, out_dtype)
                                    else:
                                        out[i] = T.cast(v - stat[0] - scale, out_dtype)
                                for i in T.vectorized(vec):
                                    y[row, col + i] = out[i]

        return main

    return _func

def _held_row(M: int, N: int, dtype: str, out_dtype: str, threads: int, vec: int):
    """Softmax of one row a block, each thread holding ``N / threads`` elements in registers.

    Thread ``t`` holds vectors ``k * threads + t``. Its loads are evict-last in L2: at
    (32768, 8192) float16 on H200 that took 248.6 us against 259.8 with the fragment
    kernel's default loads. Max and sum fold across a warp with shuffles, then across
    warps through shared memory.
    """
    steps = N // (threads * vec)
    held_n = steps * vec
    num_warps = threads // WARP_LANES

    @T.prim_func
    def main(x: T.Tensor[(M, N), dtype], y: T.Tensor[(M, N), out_dtype]):
        with T.Kernel(M, threads=threads) as row:
            tx = T.get_thread_binding()
            held = T.alloc_local((held_n,), dtype)
            vals = T.alloc_local((held_n,), "float32")
            out = T.alloc_local((held_n,), out_dtype)
            part = T.alloc_local((1,), "float32")
            warp_max = T.alloc_shared((num_warps,), "float32")
            warp_sum = T.alloc_shared((num_warps,), "float32")

            for k in T.unroll(steps):
                load16_evict_last(held, k * vec, x, (row, (k * threads + tx) * vec))
            for i in T.unroll(held_n):
                vals[i] = T.cast(held[i], "float32")

            part[0] = vals[0]
            for i in T.unroll(1, held_n):
                part[0] = T.max(part[0], vals[i])
            for stage in T.unroll(WARP_SHUFFLE_STAGES):
                part[0] = T.max(
                    part[0],
                    T.shfl_xor(part[0], T.int32(WARP_LANES // 2) >> stage, width=WARP_LANES),
                )
            if tx % WARP_LANES == 0:
                warp_max[tx // WARP_LANES] = part[0]
            T.sync_threads()
            for w in T.unroll(num_warps):
                part[0] = T.max(part[0], warp_max[w])

            for i in T.unroll(held_n):
                vals[i] = exp_shifted(vals[i], part[0])
            part[0] = vals[0]
            for i in T.unroll(1, held_n):
                part[0] = part[0] + vals[i]
            for stage in T.unroll(WARP_SHUFFLE_STAGES):
                part[0] = part[0] + T.shfl_xor(
                    part[0], T.int32(WARP_LANES // 2) >> stage, width=WARP_LANES
                )
            if tx % WARP_LANES == 0:
                warp_sum[tx // WARP_LANES] = part[0]
            T.sync_threads()
            part[0] = warp_sum[0]
            for w in T.unroll(1, num_warps):
                part[0] = part[0] + warp_sum[w]

            part[0] = 1.0 / part[0]
            for i in T.unroll(held_n):
                out[i] = T.cast(vals[i] * part[0], out_dtype)
            for k in T.unroll(steps):
                for c in T.vectorized(vec):
                    y[row, (k * threads + tx) * vec + c] = out[k * vec + c]

    return main

@functools.lru_cache(maxsize=64)
def _softmax_kernel_single(M: int, N: int, op_kind: str, dtype: str, out_dtype: str):
    """Build a single-tile softmax/log_softmax kernel (N fits in smem).

    Accepts an ``(M, N)`` input tensor. A softmax row of whole thread-wide vectors, one a
    block, runs ``_held_row``. Otherwise, when ``N`` is not a multiple of
    ``DEFAULT_ALIGNMENT``, the kernel uses element-wise ``T.if_then_else``
    loads that substitute ``-inf`` for out-of-bounds columns (kernel-side
    boundary handling), and when ``N`` is already aligned, the fast ``T.copy``
    path is used.

    softmax never reads the row again once it has exponentiated, so it goes global to
    fragment and back with nothing staged in between, dropping the shared round trip
    it would otherwise pay. log_softmax needs the row a second time, for
    ``(x - max) - log(sum)``,
    and shared memory is the cheaper of the two places to keep it -- the alternative is
    a second fragment of row width, which is what pushes a thread's slice into local
    memory.
    """
    N_padded = align_up(N, DEFAULT_ALIGNMENT)
    _needs_pad = N_padded != N
    # Compile-time Python constant used for padding; it is still cast to
    # the kernel dtype where needed inside the generated kernel.
    _neg_inf = float("-inf")
    _stages_row = op_kind == "log_softmax"
    vec = VECTOR_ACCESS_BYTES // (DataType(dtype).bits // 8)

    @tilelang.jit(out_idx=[1], compile_flags=csrc_include("streaming_load.h"))
    def _func(block_m, threads):
        # A softmax row of whole thread-wide vectors is held in registers, read
        # evict-last in L2 and reduced with warp shuffles; see ``_held_row``. On H200 it
        # ran up to 64 elements a thread (16384 columns at 256 threads); past the fragment
        # budget a thread's three held buffers would spill, so wider rows keep the tile.
        if (
            op_kind == "softmax"
            and block_m == 1
            and N % (threads * vec) == 0
            and N // threads <= FRAGMENT_ELEMS_PER_THREAD
        ):
            return _held_row(M, N, dtype, out_dtype, threads, vec)

        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            y: T.Tensor[(M, N_padded), out_dtype],
        ):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                staged = T.alloc_shared((block_m, N_padded if _stages_row else 1), dtype)
                x_f32 = T.alloc_fragment((block_m, N_padded), "float32")
                row_max = T.alloc_fragment((block_m,), "float32")
                row_sum = T.alloc_fragment((block_m,), "float32")
                row_scale = T.alloc_fragment((block_m,), "float32")

                if _stages_row:
                    if _needs_pad:
                        # Element-wise load, masked for padding columns and the row tail.
                        for i in T.serial(block_m):
                            for j in T.Parallel(N_padded):
                                staged[i, j] = T.if_then_else(
                                    T.And(pid_m * block_m + i < M, j < N),
                                    x[pid_m * block_m + i, j],
                                    T.cast(_neg_inf, dtype),
                                )
                    else:
                        T.copy(x[pid_m * block_m, 0], staged)
                    for i in T.serial(block_m):
                        for j in T.Parallel(N_padded):
                            x_f32[i, j] = T.cast(staged[i, j], "float32")
                elif _needs_pad:
                    for i in T.serial(block_m):
                        for j in T.Parallel(N_padded):
                            x_f32[i, j] = T.if_then_else(
                                T.And(pid_m * block_m + i < M, j < N),
                                T.cast(x[pid_m * block_m + i, j], "float32"),
                                T.cast(_neg_inf, "float32"),
                            )
                else:
                    for i in T.serial(block_m):
                        for j in T.Parallel(N_padded):
                            x_f32[i, j] = T.cast(x[pid_m * block_m + i, j], "float32")

                T.fill(row_max, -T.infinity("float32"))
                T.reduce_max(x_f32, row_max, dim=1, clear=False)

                for i in T.serial(block_m):
                    for j in T.Parallel(N_padded):
                        x_f32[i, j] = exp_shifted(x_f32[i, j], row_max[i])
                T.reduce_sum(x_f32, row_sum, dim=1)

                if op_kind == "softmax":
                    # One reciprocal per row, then a multiply per element.
                    for i in T.Parallel(block_m):
                        row_scale[i] = 1.0 / row_sum[i]
                    for i in T.serial(block_m):
                        for j in T.Parallel(N_padded):
                            x_f32[i, j] = x_f32[i, j] * row_scale[i]
                else:
                    for i in T.Parallel(block_m):
                        row_scale[i] = T.log(row_sum[i])
                    # (x - max) - log(sum) avoids log(0) on padding; x comes back from shared.
                    for i in T.serial(block_m):
                        for j in T.Parallel(N_padded):
                            x_f32[i, j] = (
                                T.cast(staged[i, j], "float32") - row_max[i] - row_scale[i]
                            )

                for i in T.serial(block_m):
                    for j in T.Parallel(N_padded):
                        y[pid_m * block_m + i, j] = T.cast(x_f32[i, j], out_dtype)

        return main

    return _func
```

## 来源与验证

- 来源：[tile-ai/TileOPs · softmax.py](https://github.com/tile-ai/TileOPs/blob/2c54caece3bc298b8794a364f8ed0c126fd7289d/src/tileops/_kernels/reduction/softmax.py)，commit `2c54caece3bc298b8794a364f8ed0c126fd7289d`。
- 摘录范围：L16–L16, L19–L19, L20–L20, L22–L22, L24–L24, L25–L25, L27–L38, L46–L46, L47–L47, L48–L48, L57–L183, L189–L254, L257–L369。仅选取核心及其模块内依赖，保留函数体；去除与所选入口无关的外围代码。本文是摘录与分析，非上游原文件。
- 验证：[上游测试或参考入口](https://github.com/tile-ai/TileOPs/blob/2c54caece3bc298b8794a364f8ed0c126fd7289d/tests/ops/test_softmax.py)；本轮只核查入口与源码，未执行 GPU 正确性、编译或性能测试。
- 代码版权与许可遵循[上游 LICENSE](../_licenses/tileops.txt)；代码中的原有版权声明保留。核查日期：2026-10-29。
