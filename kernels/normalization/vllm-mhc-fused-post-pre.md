# vLLM mHC：带适用边界的 post/pre 融合

将 post mapping 与下一次 pre 的 FMA 投影合并，并显式判断何时应回退到分开的实现。

## 什么时候选择

- **适用**：配置选择函数返回可用配置；hidden、线程数、split 和输出 tile 的整除关系成立，主要针对小 token 负载。
- **不适用**：绕过返回 None 的适用性判断；把 GB300 调整的参数当成所有 GPU 的最佳配置。
- **性能依据**：以下收益是源码机制分析，不代表已测得加速；没有本地 GPU 结果。

## 为什么这样设计

| 原来的问题 | 设计选择 | 收益与代价 |
| --- | --- | --- |
| 小 token 下 launch/读取占比高 | 复用 new_r 计算投影和平方和 | 降低数据往返；输出 tile 间可能重复 post 计算 |
| 分段参数必须匹配 shape | 配置函数限制 token、hidden、架构并选择 splits | 减少无效配置；覆盖范围受限 |
| 每个 shape 都调参增加编译成本 | 分桶选择少量配置 | 减少编译变体；牺牲部分单 shape 最佳性能 |

**下一步验证**：先检查分派函数选择的路径，再同时测 fused 和 fallback 的总延迟、冷编译成本与数值差异。

## 核心完整代码

入口：`mhc_fused_post_pre_split_config`, `mhc_fused_post_pre_splits`, `mhc_fused_tilelang`。包含完整适用性选择与 FMA 工厂；输出仍是 partial，外层组合见同目录 tilelang.py。

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math

from functools import cache

from vllm.platforms import current_platform

from vllm.tilelang_utils import T, tilelang, tilelang_jit

ENABLE_PDL = current_platform.is_arch_support_pdl() and current_platform.is_cuda()

_FUSED_POST_PRE_N_THR = 128

_FUSED_POST_PRE_MAX_TOKENS = 32

_FUSED_POST_PRE_SINGLE_PASS_MAX_TOKENS = 4

_FUSED_POST_PRE_SINGLE_PASS = (2, 16, 256)

@cache
def _fused_post_pre_single_pass_supported() -> bool:
    return current_platform.is_cuda() and current_platform.has_device_capability(100)

def mhc_fused_post_pre_split_config(
    num_tokens: int, hidden_size: int, hc_mult: int
) -> tuple[int, int, int] | None:
    """Pick ``(tile_n, n_splits, n_thr)`` for the fused post + pre-norm GEMM.

    Returns None when a separate post kernel followed by a split-k GEMM is the
    better choice, or when the hidden size does not divide evenly across the
    fused kernel's block. One source of truth for the dispatch decision, the
    compile key and the launch, which must agree.
    """
    if num_tokens > _FUSED_POST_PRE_MAX_TOKENS:
        return None
    tile_n, n_splits, n_thr = _FUSED_POST_PRE_SINGLE_PASS
    if (
        num_tokens <= _FUSED_POST_PRE_SINGLE_PASS_MAX_TOKENS
        and hidden_size == n_splits * n_thr
        and (hc_mult * (hc_mult + 2)) % tile_n == 0
        and _fused_post_pre_single_pass_supported()
    ):
        return tile_n, n_splits, n_thr
    n_thr = _FUSED_POST_PRE_N_THR
    if hidden_size % n_thr:
        return None
    # The projection tiles supply the parallelism a small token count cannot,
    # so the tile grows with the batch: 1.3x at one token and 1.03x at 32
    # against a split-k GEMM given the split its own estimator computes.
    if num_tokens < 16:
        tile_n, n_splits = 2, 8
    else:
        tile_n, n_splits = 6, 8
    # The kernel drops projection tiles and k-slices it cannot fill evenly.
    mix_size = hc_mult * (hc_mult + 2)
    while tile_n > 1 and mix_size % tile_n:
        tile_n -= 1
    while n_splits > 1 and hidden_size % (n_splits * n_thr):
        n_splits //= 2
    return tile_n, n_splits, n_thr

def mhc_fused_post_pre_splits(hidden_size: int, hc_mult: int) -> tuple[int, ...]:
    """Every split-k factor the fused post + pre-norm GEMM path can pick."""
    configs = (
        mhc_fused_post_pre_split_config(num_tokens, hidden_size, hc_mult)
        for num_tokens in range(1, _FUSED_POST_PRE_MAX_TOKENS + 1)
    )
    return tuple(sorted({config[1] for config in configs if config is not None}))

@tilelang_jit
def mhc_fused_tilelang(
    comb_mix,
    residual_in,
    post_mix,
    x_in,
    weight_t,
    yp_out,
    rp_out,
    residual_out,
    hc: int,
    hidden: int,
    n_out: int,
    n_thr: int = 256,
    h_blk: int = 256,
    tile_n: int = 1,
    split_k: int = 1,
) -> tilelang.JITKernel:
    """Fused mhc post-mapping + pre-norm GEMM FMA."""
    m = T.dynamic("num_tokens")
    split_k = T.dynamic("split_k")
    h = hidden
    h_blk = math.gcd(hidden, h_blk)
    h_per_split = h // split_k
    n_tiles = n_out // tile_n

    comb_mix: T.Tensor((m, hc, hc), T.float32)  # type: ignore[no-redef, valid-type]
    residual_in: T.Tensor((m, hc, h), T.bfloat16)  # type: ignore[no-redef, valid-type]
    post_mix: T.Tensor((m, hc), T.float32)  # type: ignore[no-redef, valid-type]
    x_in: T.Tensor((m, h), T.bfloat16)  # type: ignore[no-redef, valid-type]
    weight_t: T.Tensor((n_out, hc, h), T.float32)  # type: ignore[no-redef, valid-type]
    yp_out: T.Tensor((split_k, m, n_out), T.float32)  # type: ignore[no-redef, valid-type]
    rp_out: T.Tensor((split_k, m), T.float32)  # type: ignore[no-redef, valid-type]
    residual_out: T.Tensor((m, hc, h), T.bfloat16)  # type: ignore[no-redef, valid-type]

    h_iters = h_per_split // n_thr
    num_warps = n_thr // 32

    with T.Kernel(m, n_tiles, split_k, threads=n_thr) as (i_n, i_nt, i_ks):
        tid = T.get_thread_binding()
        warp_id = tid // 32
        lane = tid % 32

        s_warp = T.alloc_shared((num_warps, tile_n + 1), T.float32)
        s_post = T.alloc_shared((hc,), T.float32)
        s_comb = T.alloc_shared((hc, hc), T.float32)

        pm = T.alloc_local((hc,), T.float32)
        cm = T.alloc_local((hc, hc), T.float32)
        acc = T.alloc_local((tile_n,), T.float32)
        sqr = T.alloc_local((1,), T.float32)
        new_r = T.alloc_local((hc,), T.float32)

        T.clear(acc)
        T.clear(sqr)
        h_split_start = i_ks * h_per_split

        if ENABLE_PDL:
            T.pdl_sync()

        T.copy(post_mix[i_n, 0], s_post)
        T.copy(comb_mix[i_n, 0, 0], s_comb)

        for j in T.unroll(hc):
            pm[j] = s_post[j]
        for j in T.unroll(hc):
            for k in T.unroll(hc):
                cm[k, j] = s_comb[k, j]

        # Each thread owns h_iters elements of the k-split's h slice.
        for it in T.serial(h_iters):
            h_idx = h_split_start + it * n_thr + tid

            # Compute new residual from layer output and past residual
            for j in T.unroll(hc):
                new_r[j] = pm[j] * x_in[i_n, h_idx]
                for k in T.unroll(hc):
                    new_r[j] += cm[k, j] * residual_in[i_n, k, h_idx]

            # populate residual_out and compute sqr sum
            if i_nt == 0:
                for j in T.unroll(hc):
                    residual_out[i_n, j, h_idx] = new_r[j]
                    sqr[0] += new_r[j] * new_r[j]

            # Per-thread FMA into acc[n]
            for n in T.unroll(tile_n):
                for j in T.unroll(hc):
                    acc[n] += weight_t[i_nt * tile_n + n, j, h_idx] * new_r[j]

        for n in T.unroll(tile_n):
            acc[n] = T.warp_reduce_sum(acc[n])
        if i_nt == 0:
            sqr[0] = T.warp_reduce_sum(sqr[0])

        # Cross-warp reduce via shared mem
        if lane == 0:
            for n in T.unroll(tile_n):
                s_warp[warp_id, n] = acc[n]
            if i_nt == 0:
                s_warp[warp_id, tile_n] = sqr[0]
        T.sync_threads()

        # Warp 0 does the final cross-warp sum and writes outputs
        if warp_id == 0:
            if lane < tile_n:
                v = T.alloc_var(T.float32, init=0.0)
                for w in T.unroll(num_warps):
                    v += s_warp[w, lane]
                yp_out[i_ks, i_n, i_nt * tile_n + lane] = v

            if i_nt == 0 and lane == 0:
                v2 = T.alloc_var(T.float32, init=0.0)
                for w in T.unroll(num_warps):
                    v2 += s_warp[w, tile_n]
                rp_out[i_ks, i_n] = v2

        if ENABLE_PDL:
            T.pdl_trigger()
```

## 来源与验证

- 来源：[vllm-project/vllm · tilelang_kernels.py](https://github.com/vllm-project/vllm/blob/fa3b9d1377219247060d3c8b49eae199c9e7e154/vllm/model_executor/kernels/mhc/tilelang_kernels.py)，commit `fa3b9d1377219247060d3c8b49eae199c9e7e154`。
- 摘录范围：L4–L4, L6–L6, L8–L8, L20–L20, L21–L21, L24–L24, L63–L63, L64–L64, L70–L70, L71–L71, L74–L76, L79–L115, L131–L137, L692–L810。仅选取核心及其模块内依赖，保留函数体；去除与所选入口无关的外围代码。本文是摘录与分析，非上游原文件。
- 验证：[上游测试或参考入口](https://github.com/vllm-project/vllm/blob/fa3b9d1377219247060d3c8b49eae199c9e7e154/tests/kernels/mhc/test_mhc_kernels.py)；本轮只核查入口与源码，未执行 GPU 正确性、编译或性能测试。
- 代码版权与许可遵循[上游 LICENSE](../_licenses/vllm.txt)；代码中的原有版权声明保留。核查日期：2026-10-29。
