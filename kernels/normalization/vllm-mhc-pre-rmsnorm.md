# vLLM mHC：线程分工的 pre + RMSNorm

在线程组间分配 mixing 与 residual collapse 工作，融合 mHC pre 的后半段和 RMSNorm。

## 什么时候选择

- **适用**：vLLM 的 mHC partial、weight、residual 布局和 hc 配置；使用对应平台的线程分工与 JIT 包装。
- **不适用**：仅复制 CUDA 的线程区间到 ROCm；忽略 mix/collapse 线程与 wave 对齐关系。
- **性能依据**：以下收益是源码机制分析，不代表已测得加速；没有本地 GPU 结果。

## 为什么这样设计

| 原来的问题 | 设计选择 | 收益与代价 |
| --- | --- | --- |
| mixing 与大 hidden collapse 工作量不同 | _is_mix_thread/_is_collapse_thread 分工 | 让不同工作并行推进；需要显式协作 |
| collapse 输出随后被归一化 | 共享内存暂存，再做 RMSNorm | 减少中间 HBM；资源压力随 hidden 变化 |
| ROCm wave 与 CUDA warp 不同 | 平台条件选择 mixing 线程位置 | 避开不合适的线程分组；不是跨平台无条件同配置 |

**下一步验证**：与 SGLang 对应案例比较相同输入、舍入顺序、PDL 和 graph 设置；分别检查正确性与整段时间。

## 核心完整代码

入口：`mhc_pre_big_fuse_with_norm_tilelang`。依赖 vllm.tilelang_utils 的 tilelang_jit 和平台接口；完整保留工厂和平台辅助函数，不包含上层 warmup 注册类。 本文按默认 fused 模式解释；stats/input 分支需要 carried pre_mix 与 save_pre_mix，不能当成独立等价调用。

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math

from vllm.platforms import current_platform

from vllm.tilelang_utils import T, tilelang, tilelang_jit

ENABLE_PDL = current_platform.is_arch_support_pdl() and current_platform.is_cuda()

_MIX_ON_TRAILING_THREADS = current_platform.is_rocm()

def _is_mix_thread():
    if _MIX_ON_TRAILING_THREADS:
        return T.get_thread_binding() >= 64
    return T.get_thread_binding() < 32

def _is_collapse_thread():
    if _MIX_ON_TRAILING_THREADS:
        return T.get_thread_binding() < 64
    return T.get_thread_binding() >= 32

@tilelang_jit
def mhc_pre_big_fuse_with_norm_tilelang(
    gemm_out_mul,
    gemm_out_sqrsum,
    hc_scale,
    hc_base,
    residual,
    post_mix,
    comb_mix,
    layer_input,
    norm_weight,
    pre_mix_in,
    pre_mix_out,
    aux_out,
    hidden_size: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_eps: float,
    n_splits: int = 16,
    hc_mult: int = 4,
    gemm_last_dim: int = -1,
    use_pre_mix_in: bool = False,
    save_pre_mix: bool = False,
    rms_numel: int = 0,
    split_mode: str = "fused",
    write_aux: bool = False,
):
    # Split modes require shifted mHC: input collapse uses a carried pre-mix.
    assert split_mode in ("fused", "stats", "input")
    assert split_mode == "fused" or save_pre_mix
    num_tokens = T.dynamic("num_tokens")
    hc_mult3 = hc_mult * (2 + hc_mult)
    if gemm_last_dim < 0:
        gemm_last_dim = hc_mult3
    if rms_numel == 0:
        rms_numel = hc_mult * hidden_size
    hidden_block = math.gcd(1024, hidden_size)

    gemm_out_mul: T.Tensor[[n_splits, num_tokens, gemm_last_dim], T.float32]  # type: ignore[no-redef, valid-type]
    gemm_out_sqrsum: T.Tensor[[n_splits, num_tokens], T.float32]  # type: ignore[no-redef, valid-type]
    hc_scale: T.Tensor[[3], T.float32]  # type: ignore[no-redef, valid-type]
    hc_base: T.Tensor[[hc_mult3], T.float32]  # type: ignore[no-redef, valid-type]
    residual: T.Tensor[[num_tokens, hc_mult, hidden_size], T.bfloat16]  # type: ignore[no-redef, valid-type]
    post_mix: T.Tensor[[num_tokens, hc_mult], T.float32]  # type: ignore[no-redef, valid-type]
    comb_mix: T.Tensor[[num_tokens, hc_mult * hc_mult], T.float32]  # type: ignore[no-redef, valid-type]
    layer_input: T.Tensor[[num_tokens, hidden_size], T.bfloat16]  # type: ignore[no-redef, valid-type]
    norm_weight: T.Tensor[[hidden_size], T.bfloat16]  # type: ignore[no-redef, valid-type]

    pre_mix_in: T.Tensor[[num_tokens, hc_mult], T.float32]  # type: ignore[no-redef, valid-type]
    pre_mix_out: T.Tensor[[num_tokens, hc_mult], T.float32]  # type: ignore[no-redef, valid-type]
    aux_out: T.Tensor[[num_tokens, hidden_size], T.bfloat16]  # type: ignore[no-redef, valid-type]

    with T.Kernel(num_tokens, threads=96) as i:
        if ENABLE_PDL:
            T.pdl_sync()
        mixes_shared = T.alloc_shared(hc_mult3, T.float32)
        if split_mode != "input":
            rms = T.alloc_fragment(1, T.float32)
            mixes = T.alloc_fragment(hc_mult3, T.float32)
            T.clear(mixes)
            rms[0] = 0

            for i_split in T.serial(n_splits):
                rms[0] += gemm_out_sqrsum[i_split, i]
            rms[0] = T.rsqrt(rms[0] / rms_numel + rms_eps)
            for j in T.Parallel(hc_mult3):
                mixes[j] = 0
                for i_split in T.serial(n_splits):
                    mixes[j] += gemm_out_mul[i_split, i, j]
                mixes[j] *= rms[0]
            T.copy(mixes, mixes_shared)

        if split_mode != "input" and _is_mix_thread():
            cm = T.alloc_fragment((hc_mult, hc_mult), T.float32)
            for j in T.Parallel(hc_mult):
                if save_pre_mix:
                    pre_mix_out[i, j] = (
                        T.sigmoid(mixes_shared[j] * hc_scale[0] + hc_base[j])
                        + hc_pre_eps
                    )
                post_mix[i, j] = (
                    T.sigmoid(
                        mixes_shared[j + hc_mult] * hc_scale[1] + hc_base[j + hc_mult]
                    )
                    * hc_post_mult_value
                )
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = (
                    mixes_shared[j * hc_mult + k + hc_mult * 2] * hc_scale[2]
                    + hc_base[j * hc_mult + k + hc_mult * 2]
                )

            row_sum = T.alloc_fragment(hc_mult, T.float32)
            col_sum = T.alloc_fragment(hc_mult, T.float32)

            row_max = T.alloc_fragment(hc_mult, T.float32)
            T.reduce_max(cm, row_max, dim=1)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = T.exp(cm[j, k] - row_max[j])
            T.reduce_sum(cm, row_sum, dim=1)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = cm[j, k] / row_sum[j] + hc_sinkhorn_eps

            T.reduce_sum(cm, col_sum, dim=0)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

            # A NaN input row (e.g. CUDA-graph padding) has a NaN RMS, so every
            # logit and the whole result is NaN already. Skip the remaining
            # iterations: IEEE division takes a slow path on NaN. rms is an
            # rsqrt, so it is >= 0 unless NaN (T.isfinite does not lower on HIP).
            if rms[0] >= 0.0:
                for _ in T.serial(sinkhorn_repeat - 1):
                    T.reduce_sum(cm, row_sum, dim=1)
                    for j, k in T.Parallel(hc_mult, hc_mult):
                        cm[j, k] = cm[j, k] / (row_sum[j] + hc_sinkhorn_eps)

                    T.reduce_sum(cm, col_sum, dim=0)
                    for j, k in T.Parallel(hc_mult, hc_mult):
                        cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

            for j, k in T.Parallel(hc_mult, hc_mult):
                comb_mix[i, j * hc_mult + k] = cm[j, k]
        elif split_mode != "stats" and _is_collapse_thread():
            pre_mix_shared = T.alloc_shared(hc_mult, T.float32)
            for j in T.Parallel(hc_mult):
                if use_pre_mix_in:
                    pre_mix_shared[j] = pre_mix_in[i, j]
                elif save_pre_mix:
                    pre_mix_shared[j] = T.if_then_else(j == 0, 1.0, 0.0)
                else:
                    pre_mix_shared[j] = (
                        T.sigmoid(mixes_shared[j] * hc_scale[0] + hc_base[j])
                        + hc_pre_eps
                    )

            # Pass 1: stash unnormalized weighted-sum output in shared memory
            # as bf16 (matches the rounding that RMSNorm would see) while
            # accumulating the per-position squared sum.
            output_shared = T.alloc_shared(hidden_size, T.bfloat16)
            sumsq_per_pos = T.alloc_fragment(hidden_block, T.float32)
            T.clear(sumsq_per_pos)

            for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=2):
                xs = T.alloc_shared((hc_mult, hidden_block), T.bfloat16)
                xl = T.alloc_fragment((hc_mult, hidden_block), T.float32)
                T.copy(residual[i, 0, i0_h * hidden_block], xs)
                T.copy(xs, xl)

                ol = T.alloc_fragment(hidden_block, T.float32)
                T.clear(ol)
                if write_aux:
                    # Aux consumers (draft models) take the plain stream mean
                    # of the same residual this collapse reads.
                    aux = T.alloc_fragment(hidden_block, T.float32)
                    T.clear(aux)

                for i_hc in T.serial(hc_mult):
                    pre = pre_mix_shared[i_hc]
                    for i1_h in T.Parallel(hidden_block):
                        ol[i1_h] += pre * xl[i_hc, i1_h]
                        if write_aux:
                            aux[i1_h] += xl[i_hc, i1_h]

                if write_aux:
                    for i1_h in T.Parallel(hidden_block):
                        aux_out[i, i0_h * hidden_block + i1_h] = T.bfloat16(
                            aux[i1_h] / hc_mult
                        )

                if save_pre_mix:
                    # Keep the BF16 boundary before the delayed input RMSNorm.
                    rounded = T.alloc_fragment(hidden_block, T.bfloat16)
                    T.copy(ol, rounded)
                    for i1_h in T.Parallel(hidden_block):
                        value = T.float32(rounded[i1_h])
                        sumsq_per_pos[i1_h] += value * value
                        output_shared[i0_h * hidden_block + i1_h] = rounded[i1_h]
                else:
                    for i1_h in T.Parallel(hidden_block):
                        sumsq_per_pos[i1_h] += ol[i1_h] * ol[i1_h]
                        output_shared[i0_h * hidden_block + i1_h] = T.bfloat16(ol[i1_h])

            sumsq = T.alloc_fragment(1, T.float32)
            T.reduce_sum(sumsq_per_pos, sumsq, dim=0)
            rsqrt_norm = T.alloc_fragment(1, T.float32)
            rsqrt_norm[0] = T.rsqrt(sumsq[0] / hidden_size + norm_eps)

            # Pass 2: scale by rsqrt * norm_weight and write the result to HBM.
            for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=2):
                w_shared = T.alloc_shared(hidden_block, T.bfloat16)
                w_local = T.alloc_fragment(hidden_block, T.float32)
                T.copy(norm_weight[i0_h * hidden_block], w_shared)
                T.copy(w_shared, w_local)

                ol = T.alloc_fragment(hidden_block, T.float32)
                for i1_h in T.Parallel(hidden_block):
                    ol[i1_h] = (
                        output_shared[i0_h * hidden_block + i1_h]
                        * rsqrt_norm[0]
                        * w_local[i1_h]
                    )

                T.copy(ol, layer_input[i, i0_h * hidden_block])

        if ENABLE_PDL:
            T.pdl_trigger()
```

## 来源与验证

- 来源：[vllm-project/vllm · tilelang_kernels.py](https://github.com/vllm-project/vllm/blob/fa3b9d1377219247060d3c8b49eae199c9e7e154/vllm/model_executor/kernels/mhc/tilelang_kernels.py)，commit `fa3b9d1377219247060d3c8b49eae199c9e7e154`。
- 摘录范围：L4–L4, L6–L6, L20–L20, L21–L21, L24–L24, L29–L29, L32–L35, L38–L41, L318–L527。仅选取核心及其模块内依赖，保留函数体；去除与所选入口无关的外围代码。本文是摘录与分析，非上游原文件。
- 验证：[上游测试或参考入口](https://github.com/vllm-project/vllm/blob/fa3b9d1377219247060d3c8b49eae199c9e7e154/tests/kernels/mhc/test_mhc_kernels.py)；本轮只核查入口与源码，未执行 GPU 正确性、编译或性能测试。
- 代码版权与许可遵循[上游 LICENSE](../_licenses/vllm.txt)；代码中的原有版权声明保留。核查日期：2026-10-29。
