# SGLang DSA：稀疏 decode 的分段与合并

把选中 KV 分成多个 CTA 计算局部输出和 LSE，再按 softmax 权重合并，提高小 query 场景的并行度。

## 什么时候选择

- **适用**：因果 BF16 路径、kv_group=1；topk 整除 block_I×inner_iter，head 分组和尾维符合原实现。
- **不适用**：query 足够多时盲目增加 splits；将局部输出直接求平均。FP8 cache 需要另一个专用 partial 实现。
- **性能依据**：以下收益是源码机制分析，不代表已测得加速；没有本地 GPU 结果。

## 为什么这样设计

| 原来的问题 | 设计选择 | 收益与代价 |
| --- | --- | --- |
| decode 的 query CTA 太少 | 沿 topk 分段并行 | 增加可运行 CTA；多出临时缓冲与第二次 launch |
| 各段 softmax 分母不同 | 输出与局部 LSE 成对保存后加权合并 | 保持全局归一化；增加合并带宽 |
| 分段过细开销高 | inner_iter 决定每 CTA 连续处理多少 tile | 调节并行度和合并成本；必须满足整除条件 |

**下一步验证**：固定精度和 causal 语义，扫描 inner_iter，统计 partial+combine 总延迟；检查空段/无效索引是否产生 NaN。

## 核心完整代码

入口：`sparse_mla_fwd_decode_partial`, `sparse_mla_fwd_decode_combine`。这是一对完整核心工厂，分配 partial 张量和调用顺序见 tilelang_sparse_fwd；没有把 BF16 代码标成 FP8 通用实现。

```python
import tilelang

import tilelang.language as T

from tilelang.jit.adapter.base import (  # noqa: E402
    BaseKernelAdapter as _BaseKernelAdapter,
)

if not getattr(_BaseKernelAdapter, "_legalize_result_idx_patched", False):
    _orig_legalize = _BaseKernelAdapter._legalize_result_idx

    def _legalize_result_idx_safe(self, result_idx):
        if isinstance(result_idx, list):
            result_idx = list(result_idx)
        return _orig_legalize(self, result_idx)

    _BaseKernelAdapter._legalize_result_idx = _legalize_result_idx_safe
    _BaseKernelAdapter._legalize_result_idx_patched = True

pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}

if hasattr(tilelang.PassConfigKey, "TL_DISABLE_FAST_MATH"):
    pass_configs[tilelang.PassConfigKey.TL_DISABLE_FAST_MATH] = True
elif hasattr(tilelang.PassConfigKey, "TL_ENABLE_FAST_MATH"):
    pass_configs[tilelang.PassConfigKey.TL_ENABLE_FAST_MATH] = False

@tilelang.jit(
    out_idx=[-2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def sparse_mla_fwd_decode_partial(
    heads,
    dim,
    tail_dim,
    topk,
    *,
    kv_group=1,
    sm_scale=None,
    is_causal=True,
    block_I=64,
    inner_iter=1,
    num_stages=1,
    threads=256,
):
    """
    grid: (seq_len * REPLICATE_H, top_k / block_I / inner_iter)
    Each GPU block processes `inner_iter` consecutive KV tiles and writes one (partial_o, partial_lse) entry.
    """

    assert is_causal == True, "non-causal is not supported"
    assert kv_group == 1
    assert topk % block_I == 0
    assert topk % (block_I * inner_iter) == 0, (
        f"topk ({topk}) must be divisible by block_I * inner_iter = "
        f"{block_I} * {inner_iter}"
    )

    # log2(e) = 1.44269504
    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5 * 1.44269504
    else:
        sm_scale = sm_scale * 1.44269504

    batch = 1
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    head_kv = heads // kv_group
    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
    REPLICATE_H = (head_kv // 64) if head_kv > 64 else 1
    H_per_block = padded_H if REPLICATE_H == 1 else 64
    N_GROUPS = topk // (block_I * inner_iter)
    BI = block_I
    D = dim
    D_tail = tail_dim
    has_tail = D_tail > 0

    q_shape = [batch, seq_len, heads, dim + tail_dim]
    kv_shape = [batch, seq_len_kv, kv_group, dim + tail_dim]
    indices_shape = [batch, seq_len, kv_group, topk]
    partial_o_shape = [batch, seq_len, N_GROUPS, heads, dim]
    partial_lse_shape = [batch, seq_len, N_GROUPS, heads]
    indices_dtype = T.int32
    dtype = T.bfloat16
    accum_dtype = T.float32

    _q_in_shared = inner_iter == 1

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        KV: T.Tensor(kv_shape, dtype),
        Indices: T.Tensor(indices_shape, indices_dtype),
        Partial_O: T.Tensor(partial_o_shape, dtype),
        Partial_Lse: T.Tensor(partial_lse_shape, accum_dtype),
    ):
        with T.Kernel(seq_len * REPLICATE_H, N_GROUPS, threads=threads) as (bx, by):
            if _q_in_shared:
                Q_buf = T.alloc_shared([H_per_block, D], dtype)
                if has_tail:
                    Q_tail_buf = T.alloc_shared([H_per_block, D_tail], dtype)
            else:
                Q_buf = T.alloc_fragment([H_per_block, D], dtype)
                if has_tail:
                    Q_tail_buf = T.alloc_fragment([H_per_block, D_tail], dtype)

            KV_shared = T.alloc_shared([BI, D], dtype)
            if has_tail:
                K_tail_shared = T.alloc_shared([BI, D_tail], dtype)
            S_shared = T.alloc_shared([H_per_block, BI], dtype)
            mask = T.alloc_fragment([BI], T.bool)

            acc_o = T.alloc_fragment([H_per_block, D], accum_dtype)
            acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
            sumexp = T.alloc_fragment([H_per_block], accum_dtype)
            sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)
            alpha = T.alloc_fragment([H_per_block], accum_dtype)
            m_i = T.alloc_fragment([H_per_block], accum_dtype)
            m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)

            T.fill(acc_o, 0)
            T.fill(sumexp, 0)
            T.fill(m_i, -(2**30))

            b_i, g_i = 0, 0
            s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)
            group_i = by
            H0 = 0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64
            H1 = H0 + H_per_block

            T.copy(Q[b_i, s_i, H0:H1, :D], Q_buf)
            if has_tail:
                T.copy(Q[b_i, s_i, H0:H1, D:], Q_tail_buf)

            for k_i in T.Pipelined(inner_iter, num_stages=num_stages):
                topk_block_i = group_i * inner_iter + k_i

                for bi_i in T.Parallel(BI):
                    mask[bi_i] = Indices[b_i, s_i, g_i, topk_block_i * BI + bi_i] >= 0
                for bi_i, d_i in T.Parallel(BI, D):
                    idx = Indices[b_i, s_i, g_i, topk_block_i * BI + bi_i]
                    KV_shared[bi_i, d_i] = KV[
                        b_i, T.if_then_else(idx >= 0, idx, 0), g_i, d_i
                    ]
                if has_tail:
                    for bi_i, d_i in T.Parallel(BI, D_tail):
                        idx = Indices[b_i, s_i, g_i, topk_block_i * BI + bi_i]
                        K_tail_shared[bi_i, d_i] = KV[
                            b_i, T.if_then_else(idx >= 0, idx, 0), g_i, D + d_i
                        ]

                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.if_then_else(
                        mask[bi_i], 0, -T.infinity(acc_s.dtype)
                    )

                T.gemm(
                    Q_buf,
                    KV_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )
                if has_tail:
                    T.gemm(
                        Q_tail_buf,
                        K_tail_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullCol,
                    )

                T.copy(m_i, m_i_prev)
                T.reduce_max(acc_s, m_i, dim=1, clear=False)
                for h_i in T.Parallel(H_per_block):
                    alpha[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.exp2(
                        acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale
                    )
                T.reduce_sum(acc_s, sumexp_i, dim=1)
                for h_i in T.Parallel(H_per_block):
                    sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
                for h_i, d_i in T.Parallel(H_per_block, D):
                    acc_o[h_i, d_i] *= alpha[h_i]

                T.copy(acc_s, S_shared)
                T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullCol)

            # sumexp==0 (all masked), divide by 1 to get 0 and avoid nan
            for h_i, d_i in T.Parallel(H_per_block, D):
                acc_o[h_i, d_i] = acc_o[h_i, d_i] / T.if_then_else(
                    sumexp[h_i] == 0.0, 1.0, sumexp[h_i]
                )
            # sumexp==0 (all masked), use large negative so combine ignores this split
            for h_i in T.Parallel(H_per_block):
                sumexp[h_i] = T.if_then_else(
                    sumexp[h_i] == 0.0,
                    -(2**30),
                    T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale,
                )

            T.copy(acc_o, Partial_O[b_i, s_i, group_i, H0:H1, :])
            T.copy(sumexp, Partial_Lse[b_i, s_i, group_i, H0:H1])

    return main

@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def sparse_mla_fwd_decode_combine(
    heads,
    dim,
    topk,
    head_per_block,
    *,
    block_I=64,
    threads=256,
):
    """
    grid: (seq_len * REPLICATE_H). batch=1, kv_group=1.
    Each block does one tile of heads (e.g. 4 or 8 for decode).
    """

    assert heads % head_per_block == 0, f"head_per_block must divide heads"

    batch = 1
    seq_len = T.dynamic("seq_len")

    NI = topk // block_I
    H_per_block = head_per_block
    REPLICATE_H = heads // H_per_block

    partial_o_shape = [batch, seq_len, NI, heads, dim]
    partial_lse_shape = [batch, seq_len, NI, heads]
    o_shape = [batch, seq_len, heads, dim]
    dtype = T.bfloat16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        Partial_O: T.Tensor(partial_o_shape, dtype),
        Partial_Lse: T.Tensor(partial_lse_shape, accum_dtype),
        Output: T.Tensor(o_shape, dtype),
    ):
        with T.Kernel(seq_len * REPLICATE_H, threads=threads) as (bx,):
            shared_lse = T.alloc_shared([NI, H_per_block], accum_dtype)

            lse_max = T.alloc_fragment([H_per_block], accum_dtype)
            lse_sum = T.alloc_fragment([H_per_block], accum_dtype)
            scale = T.alloc_fragment([H_per_block, NI], accum_dtype)
            acc_o = T.alloc_fragment([H_per_block, dim], accum_dtype)

            b_i = 0
            s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)
            H0 = 0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * H_per_block
            H1 = H0 + H_per_block

            for k in T.serial(NI):
                T.copy(Partial_Lse[b_i, s_i, k, H0:H1], shared_lse[k, :])

            T.fill(lse_max, -(2**30))
            for k in T.serial(NI):
                for h_i in T.Parallel(H_per_block):
                    lse_max[h_i] = T.max(lse_max[h_i], shared_lse[k, h_i])
            T.fill(lse_sum, 0)
            for k in T.serial(NI):
                for h_i in T.Parallel(H_per_block):
                    lse_sum[h_i] = lse_sum[h_i] + T.exp2(
                        shared_lse[k, h_i] - lse_max[h_i]
                    )
            for k in T.serial(NI):
                for h_i in T.Parallel(H_per_block):
                    scale[h_i, k] = T.exp2(
                        shared_lse[k, h_i] - lse_max[h_i] - T.log2(lse_sum[h_i])
                    )

            T.fill(acc_o, 0)
            for k in T.serial(NI):
                for h_i, d_i in T.Parallel(H_per_block, dim):
                    acc_o[h_i, d_i] = acc_o[h_i, d_i] + scale[h_i, k] * Partial_O[
                        b_i, s_i, k, H0 + h_i, d_i
                    ].astype(accum_dtype)

            T.copy(acc_o, Output[b_i, s_i, H0:H1, :])

    return main
```

## 来源与验证

- 来源：[sgl-project/sglang · tilelang_kernel.py](https://github.com/sgl-project/sglang/blob/2a6c2045705bd2ac46136cece18f0d32c9c13650/python/sglang/kernels/ops/attention/dsa/tilelang_kernel.py)，commit `2a6c2045705bd2ac46136cece18f0d32c9c13650`。
- 摘录范围：L5–L5, L6–L6, L21–L23, L25–L34, L36–L39, L41–L44, L632–L814, L817–L900。仅选取核心及其模块内依赖，保留函数体；去除与所选入口无关的外围代码。本文是摘录与分析，非上游原文件。
- 验证：[上游测试或参考入口](https://github.com/sgl-project/sglang/blob/2a6c2045705bd2ac46136cece18f0d32c9c13650/test/registered/attention/unittests/dsa/test_dsa.py)；本轮只核查入口与源码，未执行 GPU 正确性、编译或性能测试。
- 代码版权与许可遵循[上游 LICENSE](../_licenses/sglang.txt)；代码中的原有版权声明保留。核查日期：2026-10-29。
