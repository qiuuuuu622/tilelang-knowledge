# SGLang QSA：prefill/decode 索引打分分工

按 query 数量区分 prefill 与 decode 的 MQA 打分组织方式，服务 QSA 索引器。

## 什么时候选择

- **适用**：QSA 的 query/key/head weight 语义与 dtype 匹配；使用同文件包装层决定分块、key splits 和输出。
- **不适用**：当作通用 attention 输出；忽略相对位置/有效长度约束；将 decode 配置原样用于大 prefill。
- **性能依据**：以下收益是源码机制分析，不代表已测得加速；没有本地 GPU 结果。

## 为什么这样设计

| 原来的问题 | 设计选择 | 收益与代价 |
| --- | --- | --- |
| prefill 与 decode 并行度不同 | 分别实现矩阵分块与 decode grid | 按负载组织工作；多一套调度配置 |
| 索引评分包含 head 权重 | 把激活与 head 聚合放在打分内部 | 减少中间写回；须保持评分定义 |
| 长 key 维限制单 CTA 工作 | decode 包装层结合设备 SM 数配置 CTA | 改善设备利用率；更多划分未必更快 |

**下一步验证**：运行 QSA indexer 测试；分别测短/长 key、少/多 query，并核对 mask 和权重后再比较分块。

## 核心完整代码

入口：`tilelang_qsa_mqa_prefill`, `tilelang_qsa_mqa_decode`。保留 HAS_TILELANG 条件块中的完整相关工厂和两个包装入口；依赖 SGLang 环境。

```python
import functools

import math

from typing import Optional

import torch

from sglang.srt.utils.common import is_hip

try:
    import tilelang
    from tilelang import language as T

    HAS_TILELANG = True
except ImportError:
    tilelang = None
    T = None
    HAS_TILELANG = False

def _validate_q(q: torch.Tensor) -> None:
    if q.ndim != 3 or q.shape[1] <= 0 or q.shape[2] <= 0:
        raise ValueError(f"QSA requires q [tokens, heads, head_dim], got {q.shape}")

def _validate_k(k: torch.Tensor) -> None:
    if k.ndim != 3 or k.shape[1] != 1 or k.shape[2] <= 0:
        raise ValueError(f"QSA MQA requires k [tokens, 1, head_dim], got {k.shape}")

def _validate_decode_inputs(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    context_lens: torch.Tensor,
) -> None:
    _validate_q(q)
    if k_cache.ndim != 4 or k_cache.shape[2] != 1:
        raise ValueError(
            "QSA decode cache must be [pages, page_size, 1, head_dim], "
            f"got {tuple(k_cache.shape)}"
        )
    if k_cache.shape[-1] != q.shape[-1]:
        raise ValueError("QSA query and key head dimensions must match")
    if page_table.ndim != 2 or page_table.shape[0] != q.shape[0]:
        raise ValueError("QSA decode page table must have one row per query")
    if context_lens.numel() != q.shape[0]:
        raise ValueError("QSA decode context lengths must have one entry per query")

_TILELANG_DTYPES = {
    torch.bfloat16: "bfloat16",
    torch.float8_e4m3fn: "float8_e4m3fn",
}

def _scoring_dtype(q: torch.Tensor, k: torch.Tensor) -> torch.dtype:
    """The dtype both GEMM operands run in: fp8 only when Q and K already are."""
    if q.dtype == torch.float8_e4m3fn or k.dtype == torch.float8_e4m3fn:
        if q.dtype != k.dtype:
            raise ValueError(
                "QSA fp8 scoring needs Q and the compressed K in the same dtype, "
                f"got q={q.dtype} k={k.dtype}"
            )
        return torch.float8_e4m3fn
    return torch.bfloat16

if HAS_TILELANG:

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        }
    )
    def _tilelang_qsa_mqa_prefill_kernel(
        heads: int,
        head_dim: int,
        block_n: int = 64,
        block_q: int = 32,
        num_stages: int = 3,
        threads: int = 512,
        dtype: str = "bfloat16",
    ):
        rows = T.dynamic("rows")
        keys = T.dynamic("keys")

        @T.prim_func
        def kernel(
            Q: T.Tensor([rows * heads, head_dim], dtype),  # type: ignore
            K: T.Tensor([keys, head_dim], dtype),  # type: ignore
            Logits: T.Tensor([rows, keys], T.float32),  # type: ignore
            Starts: T.Tensor([rows], T.int32),  # type: ignore
            Ends: T.Tensor([rows], T.int32),  # type: ignore
        ):
            with T.Kernel(T.ceildiv(rows, block_q), threads=threads) as bx:
                q_shared = T.alloc_shared([block_q * heads, head_dim], dtype)
                k_shared = T.alloc_shared([block_n, head_dim], dtype)
                scores = T.alloc_fragment([block_n, block_q * heads], T.float32)
                scores_3d = T.reshape(scores, (block_n, block_q, heads))
                reduced = T.alloc_fragment([block_n, block_q], T.float32)
                row_base = bx * block_q
                start_min = T.alloc_var(T.int32)
                end_max = T.alloc_var(T.int32)
                start_min = 2147483647
                end_max = -2147483648
                for qi in T.serial(block_q):
                    start_min = T.min(start_min, T.min(Starts[row_base + qi], keys))
                    end_max = T.max(end_max, T.min(Ends[row_base + qi], keys))

                T.copy(Q[row_base * heads, 0], q_shared)
                for ni in T.Pipelined(
                    T.ceildiv(end_max - start_min, block_n), num_stages=num_stages
                ):
                    T.copy(K[start_min + ni * block_n, 0], k_shared)
                    T.gemm(
                        k_shared,
                        q_shared,
                        scores,
                        transpose_B=True,
                        clear_accum=True,
                        policy=T.GemmWarpPolicy.FullCol,
                    )
                    for n, qi, head in T.Parallel(block_n, block_q, heads):
                        scores_3d[n, qi, head] = T.max(scores_3d[n, qi, head], 0.0)
                    T.reduce_sum(scores_3d, reduced, dim=-1, clear=True)
                    for qi, n in T.Parallel(block_q, block_n):
                        Logits[row_base + qi, start_min + ni * block_n + n] = reduced[
                            n, qi
                        ]

        return kernel

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        }
    )
    def _tilelang_qsa_mqa_prefill_relative_kernel(
        heads: int,
        head_dim: int,
        block_n: int = 64,
        block_q: int = 32,
        num_stages: int = 3,
        threads: int = 512,
        key_splits: int = 1,
        dtype: str = "bfloat16",
    ):
        """The packed prefill kernel with request-relative columns: row r's score
        for key ``Starts[r] + j`` lands in ``Logits[r, j]``, so the workspace is
        [rows, longest request] instead of [rows, all keys of the batch]. A
        second grid dimension splits each row block's key range, which keeps the
        grid full when few rows are scored."""
        rows = T.dynamic("rows")
        keys = T.dynamic("keys")
        width = T.dynamic("width")

        @T.prim_func
        def kernel(
            Q: T.Tensor([rows * heads, head_dim], dtype),  # type: ignore
            K: T.Tensor([keys, head_dim], dtype),  # type: ignore
            Logits: T.Tensor([rows, width], T.float32),  # type: ignore
            Starts: T.Tensor([rows], T.int32),  # type: ignore
            Ends: T.Tensor([rows], T.int32),  # type: ignore
        ):
            with T.Kernel(T.ceildiv(rows, block_q), key_splits, threads=threads) as (
                bx,
                by,
            ):
                q_shared = T.alloc_shared([block_q * heads, head_dim], dtype)
                k_shared = T.alloc_shared([block_n, head_dim], dtype)
                scores = T.alloc_fragment([block_n, block_q * heads], T.float32)
                scores_3d = T.reshape(scores, (block_n, block_q, heads))
                reduced = T.alloc_fragment([block_n, block_q], T.float32)
                row_base = bx * block_q
                start_min = T.alloc_var(T.int32)
                end_max = T.alloc_var(T.int32)
                start_min = 2147483647
                end_max = -2147483648
                for qi in T.serial(block_q):
                    start_min = T.min(start_min, T.min(Starts[row_base + qi], keys))
                    end_max = T.max(end_max, T.min(Ends[row_base + qi], keys))
                num_tiles = T.ceildiv(end_max - start_min, block_n)
                tiles_per_split = T.ceildiv(num_tiles, key_splits)
                tile_begin = by * tiles_per_split
                tile_count = T.max(T.min(num_tiles - tile_begin, tiles_per_split), 0)

                T.copy(Q[row_base * heads, 0], q_shared)
                for ti in T.Pipelined(tile_count, num_stages=num_stages):
                    key_base = start_min + (tile_begin + ti) * block_n
                    T.copy(K[key_base, 0], k_shared)
                    T.gemm(
                        k_shared,
                        q_shared,
                        scores,
                        transpose_B=True,
                        clear_accum=True,
                        policy=T.GemmWarpPolicy.FullCol,
                    )
                    for n, qi, head in T.Parallel(block_n, block_q, heads):
                        scores_3d[n, qi, head] = T.max(scores_3d[n, qi, head], 0.0)
                    T.reduce_sum(scores_3d, reduced, dim=-1, clear=True)
                    for qi, n in T.Parallel(block_q, block_n):
                        column = key_base + n - Starts[row_base + qi]
                        # A row block can straddle requests; keep each row to
                        # its own window.
                        if column >= 0:
                            if column < width:
                                Logits[row_base + qi, column] = reduced[n, qi]

        return kernel

    @tilelang.jit
    def _tilelang_qsa_mqa_mask_kernel(threads: int = 512, block_k: int = 4096):
        rows = T.dynamic("rows")
        keys = T.dynamic("keys")

        @T.prim_func
        def kernel(
            Logits: T.Tensor([rows, keys], T.float32),  # type: ignore
            Starts: T.Tensor([rows], T.int32),  # type: ignore
            Ends: T.Tensor([rows], T.int32),  # type: ignore
        ):
            with T.Kernel(rows, threads=threads) as bx:
                tx = T.thread_binding(0, threads, thread="threadIdx.x")
                for block in T.Pipelined(T.ceildiv(keys, block_k)):
                    for item in T.serial(block_k // threads):
                        column = block * block_k + item * threads + tx
                        if column < Starts[bx] or column >= Ends[bx]:
                            Logits[bx, column] = -T.infinity(T.float32)

        return kernel

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        }
    )
    def _tilelang_qsa_mqa_decode_kernel(
        heads: int,
        head_dim: int,
        page_size: int = 64,
        ctas_per_row: int = 1,
        min_scored_len: int = 0,
        threads: int = 128,
        dtype: str = "bfloat16",
    ):
        # The MMA layout needs 64 GEMM rows; a compressed page has full_page // ratio,
        # so pages are packed as sub-pages of one 64-row tile.
        GROUP = 64
        assert GROUP % page_size == 0, page_size
        sub_pages = GROUP // page_size
        batch = T.dynamic("batch")
        pages = T.dynamic("pages")
        max_pages = T.dynamic("max_pages")
        max_model_len = T.dynamic("max_model_len")

        @T.prim_func
        def kernel(
            Q: T.Tensor([batch, 1, heads, head_dim], dtype),  # type: ignore
            KCache: T.Tensor([pages, page_size, 1, head_dim], dtype),  # type: ignore
            PageTable: T.Tensor([batch, max_pages], T.int32),  # type: ignore
            ContextLens: T.Tensor([batch], T.int32),  # type: ignore
            Logits: T.Tensor([batch, max_model_len], T.float32),  # type: ignore
            Scale: T.float32,
        ):
            # A fixed number of CTAs per row strides over that row's groups, so the
            # grid no longer scales with the page-table width (the max context).
            with T.Kernel(batch, ctas_per_row, threads=threads) as (bx, cta):
                q_shared = T.alloc_shared([heads, head_dim], dtype)
                k_shared = T.alloc_shared([GROUP, head_dim], dtype)
                scores = T.alloc_fragment([GROUP, heads], T.float32)
                reduced = T.alloc_fragment([GROUP], T.float32)
                context_len = T.alloc_var(T.int32)
                context_len = ContextLens[bx]
                num_groups = T.ceildiv(context_len, GROUP)

                # Rows at or below min_scored_len select every block, so their
                # scores are never read.
                if context_len > min_scored_len:
                    T.copy(Q[bx, 0, :, :], q_shared)
                    for step in T.serial(T.ceildiv(num_groups, ctas_per_row)):
                        group = step * ctas_per_row + cta
                        if group < num_groups:
                            # TileLang's pipeliner rejects dynamic loops around smem
                            # copies; unroll the sub-pages at the Python level.
                            for sp in range(sub_pages):
                                if (group * sub_pages + sp) * page_size < context_len:
                                    T.copy(
                                        KCache[
                                            PageTable[bx, group * sub_pages + sp],
                                            :,
                                            0,
                                            :,
                                        ],
                                        k_shared[
                                            sp * page_size : (sp + 1) * page_size, :
                                        ],
                                    )
                            T.gemm(
                                k_shared,
                                q_shared,
                                scores,
                                transpose_B=True,
                                clear_accum=True,
                                policy=T.GemmWarpPolicy.FullCol,
                            )
                            for token, head in T.Parallel(GROUP, heads):
                                scores[token, head] = T.max(scores[token, head], 0.0)
                            T.reduce_sum(scores, reduced, dim=1, clear=True)
                            for token in T.Parallel(GROUP):
                                position = group * GROUP + token
                                if position < context_len:
                                    Logits[bx, position] = reduced[token] / Scale

        return kernel

def tilelang_qsa_mqa_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    score_scale: Optional[float] = None,
) -> torch.Tensor:
    """Validated TileLang packed prefill kernel with weights removed."""

    if not HAS_TILELANG:
        raise RuntimeError("TileLang is unavailable")
    _validate_q(q)
    _validate_k(k)
    rows, keys = q.shape[0], k.shape[0]
    if not rows or not keys:
        logits = torch.zeros((rows, keys), dtype=torch.float32, device=q.device)
        return logits.masked_fill_(
            torch.ones_like(logits, dtype=torch.bool), -float("inf")
        )
    heads, head_dim = q.shape[1:]
    block_q = max(1, 128 // heads)
    padding = (-rows) % block_q
    padded_rows = rows + padding
    # A torch.cat of the padding rows would copy the whole [rows, keys] fp32 matrix,
    # doubling the dominant prefill buffer; allocate pre-padded instead.
    logits = torch.zeros((padded_rows, keys), dtype=torch.float32, device=q.device)
    scoring_dtype = _scoring_dtype(q, k)
    q_padded = q.to(scoring_dtype).contiguous()
    starts = row_starts.to(device=q.device, dtype=torch.int32).contiguous()
    ends = row_ends.to(device=q.device, dtype=torch.int32).contiguous()
    if padding:
        q_padded = torch.cat([q_padded, q_padded.new_zeros(padding, heads, head_dim)])
        starts = torch.cat([starts, starts[-1:].expand(padding)])
        ends = torch.cat([ends, ends[-1:].expand(padding)])

    _tilelang_qsa_mqa_prefill_kernel(
        heads=heads,
        head_dim=head_dim,
        block_q=block_q,
        dtype=_TILELANG_DTYPES[scoring_dtype],
    )(
        q_padded.reshape(-1, head_dim),
        k[:, 0].to(scoring_dtype).contiguous(),
        logits,
        starts,
        ends,
    )
    # A leading-dimension slice that retains every column is already
    # contiguous, so do not copy this large matrix again when removing padding.
    logits = logits[:rows]
    logits.div_(score_scale or math.sqrt(head_dim))
    _tilelang_qsa_mqa_mask_kernel()(logits, starts[:rows], ends[:rows])
    return logits

_DECODE_CTAS_PER_ROW_LEVELS = (1, 16, 256)

@functools.lru_cache(maxsize=None)
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count

def _decode_ctas_per_row(rows: int, device: torch.device) -> int:
    wanted = -(-4 * _num_sms(device.index or 0) // max(rows, 1))
    for level in _DECODE_CTAS_PER_ROW_LEVELS:
        if level >= wanted:
            return level
    return _DECODE_CTAS_PER_ROW_LEVELS[-1]

def tilelang_qsa_mqa_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    context_lens: torch.Tensor,
    max_model_len: int,
    score_scale: Optional[float] = None,
    *,
    min_scored_len: int = 0,
) -> torch.Tensor:
    """Validated TileLang paged decode kernel with weights removed.

    With ``min_scored_len`` > 0, rows whose context length is at most that are not
    scored and the logits past each row's context length are left unset: only for
    a top-k that selects every block of a row that fits k and reads nothing past
    the length."""

    if not HAS_TILELANG:
        raise RuntimeError("TileLang is unavailable")
    _validate_decode_inputs(q, k_cache, page_table, context_lens)
    page_size = int(k_cache.shape[1])
    if page_size < 8 or 64 % page_size != 0:
        raise ValueError(
            "TileLang QSA decode requires a compressed page size of "
            f"8/16/32/64 (64-row GEMM sub-page packing), got {page_size}"
        )
    if min_scored_len > 0:
        logits = torch.empty(
            (q.shape[0], max_model_len), dtype=torch.float32, device=q.device
        )
    else:
        logits = torch.full(
            (q.shape[0], max_model_len),
            -float("inf"),
            dtype=torch.float32,
            device=q.device,
        )
    if not q.shape[0] or not max_model_len:
        return logits
    # CUDA MMA accepts an eight-wide N dimension, while ROCm MFMA requires
    # sixteen. Zero-padding preserves the weight-free head sum on both paths.
    query_heads, head_dim = q.shape[1:]
    head_alignment = 16 if is_hip() else 8
    kernel_heads = max(
        head_alignment,
        ((query_heads + head_alignment - 1) // head_alignment) * head_alignment,
    )
    scoring_dtype = _scoring_dtype(q, k_cache)
    q_kernel = q.to(scoring_dtype)
    if kernel_heads != query_heads:
        q_kernel = torch.cat(
            [
                q_kernel,
                q_kernel.new_zeros(q.shape[0], kernel_heads - query_heads, head_dim),
            ],
            dim=1,
        )
    _tilelang_qsa_mqa_decode_kernel(
        heads=kernel_heads,
        head_dim=head_dim,
        page_size=page_size,
        ctas_per_row=_decode_ctas_per_row(q.shape[0], q.device),
        min_scored_len=min_scored_len,
        dtype=_TILELANG_DTYPES[scoring_dtype],
    )(
        q_kernel.unsqueeze(1).contiguous(),
        k_cache.to(scoring_dtype).contiguous(),
        page_table.to(device=q.device, dtype=torch.int32).contiguous(),
        context_lens.to(device=q.device, dtype=torch.int32).contiguous(),
        logits,
        float(score_scale or math.sqrt(head_dim)),
    )
    return logits
```

## 来源与验证

- 来源：[sgl-project/sglang · mqa.py](https://github.com/sgl-project/sglang/blob/2a6c2045705bd2ac46136cece18f0d32c9c13650/python/sglang/srt/layers/attention/qsa/mqa.py)，commit `2a6c2045705bd2ac46136cece18f0d32c9c13650`。
- 摘录范围：L4–L4, L5–L5, L6–L6, L8–L8, L10–L10, L17–L25, L28–L30, L33–L35, L60–L77, L113–L116, L119–L128, L131–L383, L386–L438, L541–L541, L544–L546, L549–L554, L557–L629。仅选取核心及其模块内依赖，保留函数体；去除与所选入口无关的外围代码。本文是摘录与分析，非上游原文件。
- 验证：[上游测试或参考入口](https://github.com/sgl-project/sglang/blob/2a6c2045705bd2ac46136cece18f0d32c9c13650/test/registered/kernels/ops/attention/qsa/test_qsa_indexer.py)；本轮只核查入口与源码，未执行 GPU 正确性、编译或性能测试。
- 代码版权与许可遵循[上游 LICENSE](../_licenses/sglang.txt)；代码中的原有版权声明保留。核查日期：2026-10-29。
