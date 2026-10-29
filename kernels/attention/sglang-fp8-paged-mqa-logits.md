# SGLang：FP8 paged MQA 索引打分

从分页 KV cache 生成 MQA 索引分数，供后续稀疏选择使用。

## 什么时候选择

- **适用**：head_dim=128、block_size=64、Q 的第二维为 1；调用时显式 clean_logits=False，并匹配每页数据后接 FP32 scale 的字节布局。
- **不适用**：默认 clean_logits=True 会触发断言；不能把超过 seq_lens 的输出当成有效值，也不能将索引打分当成最终 attention。
- **性能依据**：以下收益是源码机制分析，不代表已测得加速；没有本地 GPU 结果。

## 为什么这样设计

| 原来的问题 | 设计选择 | 收益与代价 |
| --- | --- | --- |
| 分页 cache 不是连续序列 | 由 block table 映射逻辑位置到物理页 | 复用分页存储；增加地址计算 |
| 量化打分需要恢复尺度 | 先 ReLU(QK)，乘 head weight 求和，再乘 KV scale | 对应索引评分契约；query 侧缩放须与 weight 的构造一致 |
| 多个 head 共同决定索引分数 | 在 tile 内完成打分和 head 权重聚合 | 减少中间张量；输出语义不同于普通 QK |

**下一步验证**：先核对权重中包含的缩放、page 尾部和输出 padding；包装层未清理无效输出且关闭安全访问 lowering，调用方必须提供足够容量并屏蔽无效位置。通过后测打分加选择的总时间。

## 核心完整代码

入口：`tilelang_fp8_paged_mqa_logits`。保留工厂与包装函数。复用 SGLang 的 FP8 类型判断与分页 cache 契约。 相关 DSA 测试是邻近入口，不等于该 FP8 paged 打分路径已被覆盖。

```python
import functools

from typing import Any, Optional

import tilelang

import tilelang.language as T

import torch

from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz

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

_is_fp8_fnuz = is_fp8_fnuz()

FP8 = "float8_e4m3fnuz" if _is_fp8_fnuz else "float8_e4m3fn"

FP32 = "float32"

INT32 = "int32"

UINT8 = "uint8"

@functools.cache
def fp8_paged_mqa_logits_kernel(
    head_dim: int = 128,
    num_heads: int = 64,
    block_size: int = 64,
    clear_accum: bool = True,
    split_kv: int = 1,
) -> Any:
    N = T.symbolic("batch_size")
    L = T.symbolic("max_table_length")
    S = T.symbolic("max_seq_len")
    C = T.symbolic("num_blocks")
    B = block_size
    D = head_dim
    H = num_heads
    SK = int(split_kv)
    BLOCK_BYTES = B * (D + 4)
    SCALE_OFFSET = B * D

    assert D % 4 == 0
    assert H % 4 == 0
    assert D == 128
    assert SK >= 1

    @tilelang.jit(
        pass_configs={
            **pass_configs,
            tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
        }
    )
    def fp8_paged_mqa_logits(
        q: T.Tensor[(N, H, D), FP8],
        kvcache_u8: T.Tensor[(C, BLOCK_BYTES), UINT8],
        weight: T.Tensor[(N, H), FP32],
        seq_lens: T.Tensor[(N,), INT32],
        page_table: T.Tensor[(N, L), INT32],
        o: T.Tensor[(N, S), FP32],
    ) -> None:
        _ = N, L, S, C, D, H, B
        with T.Kernel(N * SK) as bxs:
            bx = bxs % N
            pid_split = bxs // N
            seq_len = seq_lens[bx]
            np_total = T.ceildiv(seq_len, B)
            stride = T.ceildiv(np_total, SK)
            i_start = pid_split * stride
            n_iters = T.max(0, T.min(stride, np_total - i_start))

            q_smem = T.alloc_shared((H, D), FP8)
            q_s_frag = T.alloc_fragment((H,), FP32)
            T.copy(q[bx, 0, 0], q_smem)
            T.copy(weight[bx, 0], q_s_frag)

            for j in T.Pipelined(n_iters, num_stages=2):
                i = i_start + j
                page = page_table[bx, i]
                k_smem_u8 = T.alloc_shared((1, B * D), UINT8)
                T.copy(kvcache_u8[page : page + 1, 0:SCALE_OFFSET], k_smem_u8)
                k_smem = T.view(k_smem_u8, (B, D), FP8)
                k_s_smem_u8 = T.alloc_shared((1, B * 4), UINT8)
                T.copy(
                    kvcache_u8[page : page + 1, SCALE_OFFSET:BLOCK_BYTES],
                    k_s_smem_u8,
                )
                k_s_smem = T.view(k_s_smem_u8, (B,), FP32)
                k_s_frag = T.alloc_fragment((B,), FP32)
                T.copy(k_s_smem, k_s_frag)

                logits = T.alloc_fragment((B, H), FP32)
                if not clear_accum:
                    T.fill(logits, 0.0)
                T.gemm(
                    k_smem,
                    q_smem,
                    logits,
                    transpose_A=False,
                    transpose_B=True,
                    clear_accum=clear_accum,
                )

                # post processing
                for h, j2 in T.Parallel(H, B):
                    logits[j2, h] = T.max(logits[j2, h], 0.0) * q_s_frag[h]
                logits_sum = T.alloc_fragment((B,), FP32)
                T.reduce_sum(logits, logits_sum, dim=1)
                for j2 in T.Parallel(B):
                    logits_sum[j2] *= k_s_frag[j2]
                T.copy(logits_sum, o[bx, i * B])

    return fp8_paged_mqa_logits

def tilelang_fp8_paged_mqa_logits(
    q_fp8: torch.Tensor,
    kvcache_fp8: torch.Tensor,
    weight: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    deep_gemm_metadata: Any,
    max_seq_len: int,
    clean_logits: bool = True,
) -> torch.Tensor:
    _ = deep_gemm_metadata
    batch_size, _, num_heads, head_dim = q_fp8.shape
    block_size = kvcache_fp8.shape[1]
    assert head_dim == 128, "TODO"
    assert block_size == 64, "TODO"
    assert q_fp8.shape == (batch_size, 1, num_heads, head_dim)
    assert kvcache_fp8.shape[1:] == (block_size, 1, head_dim + 4)
    assert weight.shape == (batch_size, num_heads)
    assert seq_lens.shape == (batch_size,)
    assert page_table.shape[0] == batch_size
    assert clean_logits == False

    logits = page_table.new_empty((batch_size, max_seq_len), dtype=torch.float32)

    NUM_CU = 256
    split_kv = split_kv = max(1, min(max_seq_len // block_size, NUM_CU // batch_size))
    kernel = fp8_paged_mqa_logits_kernel(
        head_dim=head_dim,
        num_heads=num_heads,
        block_size=block_size,
        clear_accum=clean_logits,
        split_kv=split_kv,
    )
    q_fp8 = q_fp8.view(batch_size, num_heads, head_dim)
    kvcache_u8 = kvcache_fp8.view(-1, block_size * (head_dim + 4))
    kernel(q_fp8, kvcache_u8, weight, seq_lens, page_table, logits)
    return logits
```

## 来源与验证

- 来源：[sgl-project/sglang · tilelang_kernel.py](https://github.com/sgl-project/sglang/blob/2a6c2045705bd2ac46136cece18f0d32c9c13650/python/sglang/kernels/ops/attention/dsa/tilelang_kernel.py)，commit `2a6c2045705bd2ac46136cece18f0d32c9c13650`。
- 摘录范围：L1–L1, L3–L3, L5–L5, L6–L6, L7–L7, L9–L9, L21–L23, L25–L34, L36–L39, L41–L44, L48–L48, L51–L51, L53–L53, L54–L54, L55–L55, L1259–L1348, L1351–L1387。仅选取核心及其模块内依赖，保留函数体；去除与所选入口无关的外围代码。本文是摘录与分析，非上游原文件。
- 验证：[上游测试或参考入口](https://github.com/sgl-project/sglang/blob/2a6c2045705bd2ac46136cece18f0d32c9c13650/test/registered/attention/unittests/dsa/test_dsa.py)；本轮只核查入口与源码，未执行 GPU 正确性、编译或性能测试。
- 代码版权与许可遵循[上游 LICENSE](../_licenses/sglang.txt)；代码中的原有版权声明保留。核查日期：2026-10-29。
