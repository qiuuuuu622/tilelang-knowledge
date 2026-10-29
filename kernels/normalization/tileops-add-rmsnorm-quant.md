# TileOPs：residual add + RMSNorm + 动态量化

在同一计算链里完成 residual 相加、归一化、amax 与 INT8/FP8 量化，并保留量化前规定的舍入位置。

## 什么时候选择

- **适用**：对应 vLLM 的数值语义，16 位输入、支持的量化类型；cluster 版本需硬件和 TileOPs 外部辅助代码支持。
- **不适用**：把融合等同于任意重排浮点操作；缺少 scale_ub、零输入、NaN 规则；无 cluster 支持时照搬跨 CTA 路径。
- **性能依据**：以下收益是源码机制分析，不代表已测得加速；没有本地 GPU 结果。

## 为什么这样设计

| 原来的问题 | 设计选择 | 收益与代价 |
| --- | --- | --- |
| 多个算子重复读取同一行 | held 路径保留数据，连续计算平方和与 amax | 减少 HBM 往返；寄存器压力大 |
| 行太宽无法在一个 CTA 内保留 | _held_cluster 选择 CTA cluster 并交换 partial | 扩大可保留行宽；依赖跨 CTA 通信 |
| 仍放不下时需要有界资源 | stream 路径三遍读取：平方和、amax、写 codes | 限制寄存器；并非一次读取的融合 |

**下一步验证**：先逐阶段对齐 residual、归一化舍入、scale 和量化 codes；再比较 held/cluster/stream 的总延迟与寄存器。

## 核心完整代码

入口：`_vec`, `_held_cluster`, `_fused_add_rms_norm_quant_kernel`, `_fused_add_rms_norm_quant_stream_kernel`。保留工厂、选择辅助函数与全部本文件宏；依赖 cluster_partials.h、streaming_load.h，不能脱离 TileOPs 仅运行此 Python 段。

```python
import functools

from typing import Optional

import tilelang

import tilelang.language as T

import torch

from tileops._csrc import csrc_include

from tileops._kernels.constants import VECTOR_ACCESS_BYTES

from tileops._utils import WARP_LANES, get_sm_count, get_sm_version

def _vec(n: int) -> int:
    """The widest access of at most 16 bytes of 16-bit elements that divides *n*."""
    return next(v for v in (VECTOR_ACCESS_BYTES // 2, 4, 2, 1) if n % v == 0)

def _held_cluster(n: int, rows: int, arch: int, sm_count: int) -> Optional[int]:
    """CTAs a row is split across in registers, or ``None`` when no cluster holds it.

    A row of more than 8192 elements splits on SM90 into the widest power-of-two cluster,
    up to the 8 CTAs every SM90 device launches, that keeps the launch to one CTA an SM and
    a CTA's share at 1024 elements or more. The split stands when a share is then at most
    4096 elements, or when the row needs a cluster anyway: a row wider than one CTA holds
    takes at least the narrowest cluster that holds it.
    """
    # Elements one CTA holds in registers at most: 64 a thread at 512 threads fit the 128
    # registers a thread has there; more spill.
    max_held = 32768
    vec = _vec(n)
    max_cluster = 8 if arch >= 90 else 1

    def divides(cluster):
        return n % (cluster * vec) == 0

    cluster = 1
    while n // cluster > max_held:
        cluster *= 2
        if cluster > max_cluster or not divides(cluster):
            return None
    if n > 8192:
        wide = cluster
        while (
            wide < max_cluster
            and rows * wide * 2 <= sm_count
            and n // (wide * 2) >= 1024
            and divides(wide * 2)
        ):
            wide *= 2
        if cluster > 1 or n // wide <= 4096:
            cluster = wide
    return cluster

def _pairwise_sum(part, row, count):
    """``part[row, :count]`` summed as a pairwise tree."""
    terms = [part[row, i] for i in range(count)]
    while len(terms) > 1:
        pairs = [terms[i] + terms[i + 1] for i in range(0, len(terms) - 1, 2)]
        terms = pairs + terms[len(pairs) * 2 :]
    return terms[0]

def _pair(buf, at):
    """The two elements of *buf* from *at*, as one packed value."""
    return buf[T.Ramp(at, 1, 2)]

def _constants(dtype, quant_dtype, scale_ub):
    """``(fp8, qmax, min_scale, bound)`` of a code dtype and amax bound."""
    fp8 = quant_dtype == "float8_e4m3fn"
    # vLLM's floor on a row's scale, so an all-zero row divides by a positive number.
    min_scale = 1.0 / (448.0 * 512.0) if fp8 else float(torch.finfo(torch.float32).eps)
    bound = float("inf") if scale_ub is None else scale_ub
    return fp8, 448.0 if fp8 else 127.0, min_scale, bound

@T.macro
def _load_inputs(a, b, at, x, residual, row, col, vec):
    """``vec`` elements of ``x`` and ``residual`` at ``(row, col)`` into ``a[at:]``, ``b[at:]``."""
    if vec * 2 == VECTOR_ACCESS_BYTES:
        # The L2 evict-last cache policy has no TileLang primitive.
        T.call_extern(  # lint: extern-target allow: streaming_load.h's evict-last load
            "handle", "tileops::load16_evict_last", T.address_of(a[at]), T.address_of(x[row, col])
        )
        T.call_extern(  # lint: extern-target allow: streaming_load.h's evict-last load
            "handle",
            "tileops::load16_evict_last",
            T.address_of(b[at]),
            T.address_of(residual[row, col]),
        )
    else:
        for i in T.vectorized(vec):
            a[at + i] = x[row, col + i]
        for i in T.vectorized(vec):
            b[at + i] = residual[row, col + i]

@T.macro
def _add(a, b, s, acc, at, vec):
    """``s = a + b`` unrounded; ``acc[0]``, ``acc[1]`` gather the squares of the even and
    odd elements."""
    for i in T.vectorized(vec):
        s[at + i] = T.cast(a[at + i], "float32") + T.cast(b[at + i], "float32")
    for i in T.unroll(vec):
        acc[i % 2] = T.ieee_fmaf(s[at + i], s[at + i], acc[i % 2])

@T.macro
def _store_sum(s, a, at, residual_out, row, col, vec, dtype):
    """``s[at:at + vec]`` rounded into ``a`` and written to ``residual_out[row, col:]``."""
    for i in T.vectorized(vec):
        a[at + i] = T.cast(s[at + i], dtype)
    for i in T.vectorized(vec):
        residual_out[row, col + i] = a[at + i]

@T.macro
def _normalize(s, w, y, peak, at, wat, rrms, vec, dtype, fp8):
    """``y = T(T(s * rrms) * w)``; ``peak`` keeps each lane's largest ``|y|``, skipping a NaN.

    With FP8 codes a NaN ``y`` is held as +inf, which the codes turn into +448 as vLLM's
    ``fminf`` before ``fmaxf`` does.
    """
    inf = T.cast(T.float32(float("inf")), dtype)
    if vec % 2 == 0:
        for j in T.unroll(vec // 2):
            # The packed multiply rounds as the float32 product rounded once does.
            p = T.mul2(
                T.cast(_pair(s, at + 2 * j) * T.Broadcast(rrms, 2), f"{dtype}x2"),
                _pair(w, wat + 2 * j),
            )
            peak[T.Ramp(0, 1, 2)] = T.max2(_pair(peak, 0), T.abs2(p))
            y[T.Ramp(at + 2 * j, 1, 2)] = T.min2(p, T.Broadcast(inf, 2)) if fp8 else p
    else:
        p1 = T.cast(s[at] * rrms, dtype) * w[wat]
        peak[T.Ramp(0, 1, 2)] = T.max2(_pair(peak, 0), T.abs2(T.Broadcast(p1, 2)))
        y[at] = T.if_then_else(T.isnan(T.cast(p1, "float32")), inf, p1) if fp8 else p1

@T.macro
def _codes(y, at, wide, codes, inv_scale, vec, quant_dtype, fp8, nan_clamp):
    """Codes of ``y[at:at + vec] * inv_scale`` into ``codes``; ``nan_clamp`` makes a NaN
    product +448."""
    for i in T.vectorized(vec):
        wide[i] = T.cast(y[at + i], "float32") * inv_scale
    if fp8:
        if nan_clamp:
            for i in T.vectorized(vec):
                wide[i] = T.min(wide[i], T.float32(448.0))
        # Saturates finite values and infinities to +-448.
        for i in T.vectorized(vec):
            codes[i] = T.cast(wide[i], quant_dtype)
    else:
        # Round half to even and saturate; a NaN becomes 0, as vLLM's cvt.rni.sat makes it.
        # Adding 1.5 * 2^23 rounds a value within +-2^22 half to even into the low bits of
        # the sum, whose low byte is then the int8 code.
        for i in T.unroll(vec):
            codes[i] = T.cast(
                T.reinterpret(
                    T.if_then_else(T.isnan(wide[i]), T.float32(0), T.clamp(wide[i], -128.0, 127.0))
                    + T.float32(12582912.0),
                    "int32",
                ),
                quant_dtype,
            )

@T.macro
def _warp_sum(acc):
    """``acc[0] + acc[1]`` summed over the warp into ``acc[0]`` of every lane."""
    acc[0] += acc[1]
    for k in T.unroll(WARP_LANES.bit_length() - 1):
        acc[0] += T.shfl_xor(acc[0], T.shift_left(1, k))

@T.macro
def _block_sum(acc, part, tx, warps):
    """The warps' ``acc[0]`` summed over the CTA into ``acc[0]`` of every thread."""
    if warps > 1:
        if tx % WARP_LANES == 0:
            part[0, tx // WARP_LANES] = acc[0]
        T.sync_threads()
        acc[0] = _pairwise_sum(part, 0, warps)

@T.macro
def _block_amax_bits(peak, bits, part, tx, warps):
    """The largest lane of ``peak`` over the CTA, as int32 bits, into ``bits[0]``.

    A non-negative float without NaN orders as its int32 bits, which a warp folds with one
    ``redux.sync``.
    """
    bits[0] = T.warp_reduce_max(
        T.reinterpret(T.max(T.cast(peak[0], "float32"), T.cast(peak[1], "float32")), "int32")
    )
    if warps > 1:
        if tx % WARP_LANES == 0:
            part[1, tx // WARP_LANES] = T.reinterpret(bits[0], "float32")
        T.sync_threads()
        bits[0] = T.warp_reduce_max(T.reinterpret(part[1, tx % WARP_LANES % warps], "int32"))

@functools.lru_cache(maxsize=32)
def _fused_add_rms_norm_quant_kernel(M, N, eps, dtype, quant_dtype, scale_ub, vec, cluster):
    """Return the factory for ``M`` rows held in registers, each split across ``cluster``
    CTAs.

    The tensors are viewed as ``(M * cluster, N // cluster)``, one CTA a row of the view.
    ``vec`` divides a CTA's share; an access past the share's end is skipped whole.
    """
    fp8, qmax, min_scale, bound = _constants(dtype, quant_dtype, scale_ub)
    chunk = N // cluster

    @tilelang.jit(
        out_idx=[3, 4, 5], compile_flags=csrc_include("cluster_partials.h", "streaming_load.h")
    )
    def _func(threads):
        steps = -(-chunk // (threads * vec))
        held = steps * vec
        warps = threads // WARP_LANES
        whole = chunk % (threads * vec) == 0

        @T.macro
        def exchange(peers, at, received, tx, rank, value):
            """Send *value* to slot ``peers[at, rank]`` of every peer CTA, then wait for
            theirs."""
            if tx < cluster and tx != rank:
                T.call_extern(  # lint: extern-target allow: st.async to a peer CTA has no TileLang primitive
                    "handle",
                    "tileops::send_partial",
                    T.address_of(peers[at, rank]),
                    T.address_of(received[at]),
                    tx,
                    value,
                )
            T.mbarrier_wait_parity(received[at], 0)

        @T.macro
        def codes_pass(cta, tx, a, wide, codes, q, inv_scale, nan_clamp):
            for k in T.unroll(steps):
                base = (k * threads + tx) * vec
                if whole or base < chunk:
                    _codes(a, k * vec, wide, codes, inv_scale, vec, quant_dtype, fp8, nan_clamp)
                    for i in T.vectorized(vec):
                        q[cta, base + i] = codes[i]

        @T.prim_func
        def main(
            x: T.Tensor[(M * cluster, chunk), dtype],
            residual: T.Tensor[(M * cluster, chunk), dtype],
            weight: T.Tensor[(cluster, chunk), dtype],
            q: T.Tensor[(M * cluster, chunk), quant_dtype],
            scale: T.Tensor[(M, 1), T.float32],
            residual_out: T.Tensor[(M * cluster, chunk), dtype],
        ):
            with (
                T.Kernel(M, threads=threads)
                if cluster == 1
                else T.ClusterKernel(M * cluster, cluster_dims=cluster, threads=threads)
            ) as cta:
                tx = T.get_thread_binding()
                lane = tx % WARP_LANES
                rank = cta % cluster
                a = T.alloc_local([held], dtype)
                b = T.alloc_local([held], dtype)
                w = T.alloc_local([held], dtype)
                s = T.alloc_local([held], "float32")
                wide = T.alloc_local([vec], "float32")
                codes = T.alloc_local([vec], quant_dtype)
                acc = T.alloc_local([2], "float32")
                peak = T.alloc_local([2], dtype)
                bits = T.alloc_local([1], "int32")
                # One slot a warp for each fold, so the folds need no barrier between them.
                part = T.alloc_shared([2, warps], "float32")
                if cluster > 1:
                    # One slot a peer CTA for each fold; a CTA keeps its own in registers.
                    peers = T.alloc_shared([2, cluster], "float32")
                    received = T.alloc_barrier([1, 1])
                    if tx == 0:
                        T.mbarrier_arrive_expect_tx(received[0], 4 * (cluster - 1))
                        T.mbarrier_arrive_expect_tx(received[1], 4 * (cluster - 1))
                    # Peers may send once this CTA's barriers are initialized.
                    T.cluster_arrive_relaxed()

                for k in T.unroll(steps):
                    base = (k * threads + tx) * vec
                    if whole or base < chunk:
                        _load_inputs(a, b, k * vec, x, residual, cta, base, vec)
                        for i in T.vectorized(vec):
                            w[k * vec + i] = weight[rank, base + i]

                acc[0] = T.float32(0)
                acc[1] = T.float32(0)
                for k in T.unroll(steps):
                    base = (k * threads + tx) * vec
                    if whole or base < chunk:
                        _add(a, b, s, acc, k * vec, vec)
                _warp_sum(acc)
                # Written while the warp's sum waits for the other warps'.
                for k in T.unroll(steps):
                    base = (k * threads + tx) * vec
                    if whole or base < chunk:
                        _store_sum(s, a, k * vec, residual_out, cta, base, vec, dtype)
                _block_sum(acc, part, tx, warps)
                if cluster > 1:
                    T.cluster_wait()
                    exchange(peers, 0, received, tx, rank, acc[0])
                    # Every CTA folds the same partials in the same lanes, so all take one
                    # bit-identical sum.
                    acc[0] = T.warp_reduce_sum(
                        T.if_then_else(
                            lane == rank,
                            acc[0],
                            T.if_then_else(
                                lane < cluster, peers[0, T.min(lane, cluster - 1)], T.float32(0)
                            ),
                        )
                    )
                rrms = T.rsqrt(acc[0] / T.float32(N) + eps)

                peak[0] = T.cast(T.float32(0), dtype)
                peak[1] = T.cast(T.float32(0), dtype)
                for k in T.unroll(steps):
                    base = (k * threads + tx) * vec
                    if whole or base < chunk:
                        _normalize(s, w, a, peak, k * vec, k * vec, rrms, vec, dtype, fp8)
                _block_amax_bits(peak, bits, part, tx, warps)
                if cluster > 1:
                    exchange(peers, 1, received, tx, rank, T.reinterpret(bits[0], "float32"))
                    bits[0] = T.warp_reduce_max(
                        T.if_then_else(
                            T.And(lane < cluster, lane != rank),
                            T.reinterpret(peers[1, T.min(lane, cluster - 1)], "int32"),
                            bits[0],
                        )
                    )

                row_max = T.min(T.reinterpret(bits[0], "float32"), T.float32(bound))
                row_scale = T.max(row_max / qmax, T.float32(min_scale))
                inv_scale = T.float32(1.0) / row_scale
                if tx == 0 and rank == 0:
                    scale[cta // cluster, 0] = row_scale
                # Uniform across the row: only an infinite scale makes NaN products.
                if fp8 and inv_scale == T.float32(0):
                    codes_pass(cta, tx, a, wide, codes, q, inv_scale, True)
                else:
                    codes_pass(cta, tx, a, wide, codes, q, inv_scale, False)

        return main

    return _func

@functools.lru_cache(maxsize=32)
def _fused_add_rms_norm_quant_stream_kernel(M, N, eps, dtype, quant_dtype, scale_ub, vec):
    """Return the factory for ``M`` rows, one CTA a row, read in tiles three times.

    The first read sums the squares and writes ``residual_out``, the second finds the
    absolute maximum, the third writes the codes, walking the tiles back from the row's end
    so the tiles read last are read first, while they are likeliest in L2.
    """
    fp8, qmax, min_scale, bound = _constants(dtype, quant_dtype, scale_ub)

    @tilelang.jit(out_idx=[3, 4, 5], compile_flags=csrc_include("streaming_load.h"))
    def _func(threads, accesses):
        tile = threads * accesses * vec
        tiles = -(-N // tile)
        warps = threads // WARP_LANES
        held = accesses * vec

        @T.prim_func
        def main(
            x: T.Tensor[(M, N), dtype],
            residual: T.Tensor[(M, N), dtype],
            weight: T.Tensor[(N,), dtype],
            q: T.Tensor[(M, N), quant_dtype],
            scale: T.Tensor[(M, 1), T.float32],
            residual_out: T.Tensor[(M, N), dtype],
        ):
            with T.Kernel(M, threads=threads) as row:
                tx = T.get_thread_binding()
                a = T.alloc_local([held], dtype)
                b = T.alloc_local([held], dtype)
                w = T.alloc_local([vec], dtype)
                s = T.alloc_local([held], "float32")
                wide = T.alloc_local([vec], "float32")
                codes = T.alloc_local([vec], quant_dtype)
                acc = T.alloc_local([2], "float32")
                peak = T.alloc_local([2], dtype)
                bits = T.alloc_local([1], "int32")
                part = T.alloc_shared([2, warps], "float32")

                acc[0] = T.float32(0)
                acc[1] = T.float32(0)
                for t in T.serial(tiles):
                    for k in T.unroll(accesses):
                        col = t * tile + (k * threads + tx) * vec
                        if col < N:
                            _load_inputs(a, b, k * vec, x, residual, row, col, vec)
                    for k in T.unroll(accesses):
                        col = t * tile + (k * threads + tx) * vec
                        if col < N:
                            _add(a, b, s, acc, k * vec, vec)
                            _store_sum(s, a, k * vec, residual_out, row, col, vec, dtype)
                _warp_sum(acc)
                _block_sum(acc, part, tx, warps)
                rrms = T.rsqrt(acc[0] / T.float32(N) + eps)

                peak[0] = T.cast(T.float32(0), dtype)
                peak[1] = T.cast(T.float32(0), dtype)
                for t in T.serial(tiles):
                    for k in T.unroll(accesses):
                        col = t * tile + (k * threads + tx) * vec
                        if col < N:
                            _load_inputs(a, b, k * vec, x, residual, row, col, vec)
                    for k in T.unroll(accesses):
                        col = t * tile + (k * threads + tx) * vec
                        if col < N:
                            _add(a, b, s, acc, k * vec, vec)
                            for i in T.vectorized(vec):
                                w[i] = weight[col + i]
                            _normalize(s, w, a, peak, k * vec, 0, rrms, vec, dtype, fp8)
                _block_amax_bits(peak, bits, part, tx, warps)

                row_max = T.min(T.reinterpret(bits[0], "float32"), T.float32(bound))
                row_scale = T.max(row_max / qmax, T.float32(min_scale))
                inv_scale = T.float32(1.0) / row_scale
                if tx == 0:
                    scale[row, 0] = row_scale
                for t_back in T.serial(tiles):
                    t = tiles - 1 - t_back
                    for k in T.unroll(accesses):
                        col = t * tile + (k * threads + tx) * vec
                        if col < N:
                            _load_inputs(a, b, k * vec, x, residual, row, col, vec)
                    for k in T.unroll(accesses):
                        col = t * tile + (k * threads + tx) * vec
                        if col < N:
                            _add(a, b, s, acc, k * vec, vec)
                            for i in T.vectorized(vec):
                                w[i] = weight[col + i]
                            _normalize(s, w, a, peak, k * vec, 0, rrms, vec, dtype, fp8)
                            # Uniform across the row: only an infinite scale makes NaN
                            # products.
                            if fp8 and inv_scale == T.float32(0):
                                _codes(
                                    a, k * vec, wide, codes, inv_scale, vec, quant_dtype, fp8, True
                                )
                            else:
                                _codes(
                                    a, k * vec, wide, codes, inv_scale, vec, quant_dtype, fp8, False
                                )
                            for i in T.vectorized(vec):
                                q[row, col + i] = codes[i]

        return main

    return _func
```

## 来源与验证

- 来源：[tile-ai/TileOPs · fused_add_rms_norm_quant.py](https://github.com/tile-ai/TileOPs/blob/2c54caece3bc298b8794a364f8ed0c126fd7289d/src/tileops/_kernels/norm/fused_add_rms_norm_quant.py)，commit `2c54caece3bc298b8794a364f8ed0c126fd7289d`。
- 摘录范围：L15–L15, L16–L16, L18–L18, L19–L19, L20–L20, L22–L22, L23–L23, L30–L30, L35–L37, L40–L74, L77–L83, L86–L88, L91–L97, L100–L118, L121–L128, L131–L137, L140–L160, L163–L188, L191–L196, L199–L206, L209–L223, L226–L374, L377–L481。仅选取核心及其模块内依赖，保留函数体；去除与所选入口无关的外围代码。本文是摘录与分析，非上游原文件。
- 验证：[上游测试或参考入口](https://github.com/tile-ai/TileOPs/blob/2c54caece3bc298b8794a364f8ed0c126fd7289d/tests/ops/test_fused_add_rms_norm_per_token_quant.py)；本轮只核查入口与源码，未执行 GPU 正确性、编译或性能测试。
- 代码版权与许可遵循[上游 LICENSE](../_licenses/tileops.txt)；代码中的原有版权声明保留。核查日期：2026-10-29。
