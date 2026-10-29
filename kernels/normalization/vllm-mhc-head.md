# vLLM mHC：两遍扫描的 head 融合

第一遍归约 RMS 和投影，第二遍应用 sigmoid 门控并合并 residual 通道，直接输出 head 输入。

## 什么时候选择

- **适用**：mHC head 的 BF16 residual、FP32 fn/scale/base，隐藏维分块遵循 gcd 规则。
- **不适用**：需要单独导出全部中间 mixing 张量；把两遍扫描描述为输入只读取一次。
- **性能依据**：以下收益是源码机制分析，不代表已测得加速；没有本地 GPU 结果。

## 为什么这样设计

| 原来的问题 | 设计选择 | 收益与代价 |
| --- | --- | --- |
| 多个小中间张量导致额外 launch | 同一核完成投影、RMS、sigmoid 与输出 | 减少中间写回；逻辑更复杂 |
| 全行常驻寄存器成本高 | 分块累积 reducer，再扫描应用门控 | 限制片上占用；输入读取两遍 |
| 跨线程归约必须一致 | T.alloc_reducer 维护平方和与多路投影 | 统一归约；浮点结合顺序可能变化 |

**下一步验证**：分别测两遍读取成本和省下的 launch，覆盖 hidden 分块、极端输入与小 token；对齐参考舍入顺序。

## 核心完整代码

入口：`hc_head_fuse_tilelang`。保留核心工厂；依赖 vLLM JIT/平台支持，外层分配与调度见上游包装。

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math

from vllm.platforms import current_platform

from vllm.tilelang_utils import T, tilelang, tilelang_jit

ENABLE_PDL = current_platform.is_arch_support_pdl() and current_platform.is_cuda()

@tilelang_jit
def hc_head_fuse_tilelang(
    residual,
    fn,
    hc_scale,
    hc_base,
    out,
    hidden_size: int,
    rms_eps: float,
    hc_eps: float,
    hc_mult: int = 4,
    n_thr: int = 128,
    h_blk: int = 1024,
):
    """Two-pass fused kernel for hc_head.

    Pass 1: accumulate per-token squared sum and hc_mult dot-products
            (projections onto fn rows) using cross-thread reducers.
    Pass 2: apply sigmoid-gated weighted sum of residual channels to output.

    Avoids materialising mixes / rsqrt / pre tensors to global memory.
    """
    num_tokens = T.dynamic("num_tokens")
    hc_dim = hc_mult * hidden_size
    h_block = math.gcd(h_blk, hidden_size)
    n_h = hidden_size // h_block

    residual: T.Tensor[[num_tokens, hc_mult, hidden_size], T.bfloat16]  # type: ignore[no-redef,valid-type]
    fn: T.Tensor[[hc_mult, hc_dim], T.float32]  # type: ignore[no-redef,valid-type]
    hc_scale: T.Tensor[[1], T.float32]  # type: ignore[no-redef,valid-type]
    hc_base: T.Tensor[[hc_mult], T.float32]  # type: ignore[no-redef,valid-type]
    out: T.Tensor[[num_tokens, hidden_size], T.bfloat16]  # type: ignore[no-redef,valid-type]

    with T.Kernel(num_tokens, threads=n_thr) as i:
        if ENABLE_PDL:
            T.pdl_sync()

        # ------------------------------------------------------------------
        # Pass 1 – for each residual channel m_c and h_block:
        #   • accumulate squared sum (for RMS norm denominator)
        #   • accumulate hc_mult dot-products with fn rows
        # ------------------------------------------------------------------
        sqrsum_r = T.alloc_reducer((1,), T.float32, replication="all")
        mixes_r = T.alloc_reducer((hc_mult,), T.float32, replication="all")
        T.fill(sqrsum_r, 0.0)
        T.fill(mixes_r, 0.0)

        for m_c in T.serial(hc_mult):
            for i_h in T.serial(n_h):
                x_local = T.alloc_fragment(h_block, T.float32)
                T.copy(residual[i, m_c, i_h * h_block], x_local)

                for k in T.Parallel(h_block):
                    sqrsum_r[0] += x_local[k] * x_local[k]

                for m_m in T.unroll(hc_mult):
                    fn_local = T.alloc_fragment(h_block, T.float32)
                    T.copy(fn[m_m, m_c * hidden_size + i_h * h_block], fn_local)
                    for k in T.Parallel(h_block):
                        mixes_r[m_m] += x_local[k] * fn_local[k]

        T.finalize_reducer(sqrsum_r)
        T.finalize_reducer(mixes_r)

        # ------------------------------------------------------------------
        # Compute pre_mix = sigmoid(mix * rsqrt * scale + base) + eps
        # ------------------------------------------------------------------
        pre_mix_shared = T.alloc_shared(hc_mult, T.float32)
        rsqrt_val = T.alloc_fragment(1, T.float32)
        rsqrt_val[0] = T.rsqrt(sqrsum_r[0] / hc_dim + rms_eps)
        for m in T.Parallel(hc_mult):
            pre_mix_shared[m] = (
                T.sigmoid(mixes_r[m] * rsqrt_val[0] * hc_scale[0] + hc_base[m]) + hc_eps
            )

        # ------------------------------------------------------------------
        # Pass 2 – apply_mix: pipelined weighted sum over residual channels
        # ------------------------------------------------------------------
        for i0_h in T.Pipelined(n_h, num_stages=2):
            xs = T.alloc_shared((hc_mult, h_block), T.bfloat16)
            xl = T.alloc_fragment((hc_mult, h_block), T.float32)
            T.copy(residual[i, 0, i0_h * h_block], xs, disable_tma=True)
            T.copy(xs, xl)

            ol = T.alloc_fragment(h_block, T.float32)
            T.clear(ol)
            for i_hc in T.serial(hc_mult):
                pre = pre_mix_shared[i_hc]
                for i1_h in T.Parallel(h_block):
                    ol[i1_h] += pre * xl[i_hc, i1_h]

            T.copy(ol, out[i, i0_h * h_block], disable_tma=True)

        if ENABLE_PDL:
            T.pdl_trigger()
```

## 来源与验证

- 来源：[vllm-project/vllm · tilelang_kernels.py](https://github.com/vllm-project/vllm/blob/fa3b9d1377219247060d3c8b49eae199c9e7e154/vllm/model_executor/kernels/mhc/tilelang_kernels.py)，commit `fa3b9d1377219247060d3c8b49eae199c9e7e154`。
- 摘录范围：L4–L4, L6–L6, L20–L20, L21–L21, L24–L24, L1045–L1139。仅选取核心及其模块内依赖，保留函数体；去除与所选入口无关的外围代码。本文是摘录与分析，非上游原文件。
- 验证：[上游测试或参考入口](https://github.com/vllm-project/vllm/blob/fa3b9d1377219247060d3c8b49eae199c9e7e154/tests/kernels/mhc/test_mhc_kernels.py)；本轮只核查入口与源码，未执行 GPU 正确性、编译或性能测试。
- 代码版权与许可遵循[上游 LICENSE](../_licenses/vllm.txt)；代码中的原有版权声明保留。核查日期：2026-10-29。
