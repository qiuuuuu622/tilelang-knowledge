# SGLang mHC：post 与下一次 pre 投影融合

在生成新 residual 的同时计算下一阶段投影和平方和 partial，复用刚算出的值。

## 什么时候选择

- **适用**：连续 mHC 层、BF16 residual 与 FP32 投影权重，调用方消费 split partial 并执行后续归约。
- **不适用**：把 partial 输出当成最终 mixing；token 很多时默认 FMA 优于矩阵 GEMM；忽略融合路径的舍入语义。
- **性能依据**：以下收益是源码机制分析，不代表已测得加速；没有本地 GPU 结果。

## 为什么这样设计

| 原来的问题 | 设计选择 | 收益与代价 |
| --- | --- | --- |
| post 的结果马上被 pre 再读 | 线程内 new residual 直接参与 FMA 和平方和 | 减少重复读取；与先写 BF16 再读可能存在舍入差异 |
| 小 batch 并行度不足 | token×mix-output tile×hidden split 的 grid | 扩大并行度；重复计算与 partial 开销上升 |
| 重复写 residual 会竞争 | 指定输出 tile 负责 residual/平方和写入 | 避免冗余存储；分支条件是正确性契约 |

**下一步验证**：扫 token、hidden 与 split_k，对比 unfused 参考的误差和端到端时间；同时检查投影计算是否已经比额外访存更贵。

## 核心完整代码

入口：`mhc_fused_post_pre_fma_tilelang`。保留完整 FMA 工厂与 lazy 初始化。调度、partial 归并和后续 Sinkhorn 在同文件外层包装，不能省略。

```python
import functools

import importlib

import threading

from typing import Tuple

from sglang.kernels.jit.utils import is_arch_support_pdl

_real_tilelang = None

_tilelang_load_lock = threading.Lock()

class _LazyTilelangAttr:
    def __init__(self, path: Tuple[str, ...] = ()):
        self.path = path

    def __getattr__(self, name):
        return _LazyTilelangAttr((*self.path, name))

    def __call__(self, *_args, **_kwargs):
        return _LazyTilelangAttr(self.path)

def _resolve_lazy_tilelang_value(value):
    if isinstance(value, _LazyTilelangAttr):
        obj = _load_tilelang()
        for name in value.path:
            obj = getattr(obj, name)
        return obj
    if isinstance(value, dict):
        return {
            _resolve_lazy_tilelang_value(k): _resolve_lazy_tilelang_value(v)
            for k, v in value.items()
        }
    # Keep list/tuple support so future TileLang jit kwargs such as out_idx=[...]
    # can use lazy TileLang enum values without changing the proxy.
    if isinstance(value, list):
        return [_resolve_lazy_tilelang_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_resolve_lazy_tilelang_value(v) for v in value)
    return value

def _load_tilelang():
    global _real_tilelang, _real_T, tilelang, T
    if _real_tilelang is None:
        with _tilelang_load_lock:
            if _real_tilelang is None:
                try:
                    new_tilelang = importlib.import_module("tilelang")
                    new_T = importlib.import_module("tilelang.language")
                except ImportError as exc:
                    raise RuntimeError(
                        "tilelang is not installed; this kernel cannot run on the current platform"
                    ) from exc
                new_tilelang.set_log_level("WARNING")
                tilelang = new_tilelang
                T = new_T
                _real_T = new_T
                _real_tilelang = new_tilelang
    return _real_tilelang

class _LazyTilelang:
    PassConfigKey = _LazyTilelangAttr(("PassConfigKey",))
    layout = _LazyTilelangAttr(("layout",))

    def jit(self, func=None, **jit_kwargs):
        def decorate(fn):
            compiled = None
            compile_lock = threading.Lock()

            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                nonlocal compiled
                if compiled is None:
                    with compile_lock:
                        if compiled is None:
                            real_tilelang = _load_tilelang()
                            real_kwargs = _resolve_lazy_tilelang_value(jit_kwargs)
                            compiled = real_tilelang.jit(**real_kwargs)(fn)
                return compiled(*args, **kwargs)

            return wrapper

        if callable(func):
            return decorate(func)
        return decorate

    def __getattr__(self, name):
        return _LazyTilelangAttr((name,))

tilelang = _LazyTilelang()

T = _LazyTilelangAttr()

@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
    },
)
def mhc_fused_post_pre_fma_tilelang(
    prev_comb_mix,
    prev_residual,
    prev_post_mix,
    hidden_in,
    pre_fn,
    mixes_partial_out,
    sqrsum_partial_out,
    cur_residual_out,
    hc: int,
    hidden_size: int,
    num_mix_outputs: int,
    n_thr: int = 256,
    tile_mix_outputs: int = 1,
    split_k: int = 1,
):
    num_tokens = T.dynamic("num_tokens")
    split_k = T.dynamic("split_k")

    hidden_per_split = (hidden_size + split_k - 1) // split_k
    num_mix_output_tiles = (num_mix_outputs + tile_mix_outputs - 1) // tile_mix_outputs

    prev_comb_mix: T.Tensor((num_tokens, hc, hc), T.float32)
    prev_residual: T.Tensor((num_tokens, hc, hidden_size), T.bfloat16)
    prev_post_mix: T.Tensor((num_tokens, hc), T.float32)
    hidden_in: T.Tensor((num_tokens, hidden_size), T.bfloat16)
    pre_fn: T.Tensor((num_mix_outputs, hc, hidden_size), T.float32)

    mixes_partial_out: T.Tensor((split_k, num_tokens, num_mix_outputs), T.float32)
    sqrsum_partial_out: T.Tensor((split_k, num_tokens), T.float32)
    cur_residual_out: T.Tensor((num_tokens, hc, hidden_size), T.bfloat16)

    hidden_iters_per_thread = (hidden_per_split + n_thr - 1) // n_thr
    num_warps = n_thr // 32

    ENABLE_PDL = is_arch_support_pdl()

    # CTA assignment:
    #   token_idx           : this CTA handles one token.
    #   mix_output_tile_idx : this CTA handles a small tile of mix output columns.
    #                          For HC=4, num_mix_outputs = 24:
    #                            [0:4]   -> pre logits
    #                            [4:8]   -> post logits
    #                            [8:24]  -> comb logits
    #   hidden_split_idx    : this CTA handles one split of the hidden dimension.
    #
    # Thread assignment inside one CTA:
    #   Each thread owns several hidden positions in this hidden split:
    #     hidden_idx = hidden_split_start + hidden_iter * n_thr + thread_idx
    #
    # For each owned hidden_idx, the thread computes:
    #   1. post result: cur_residual[token, :, hidden_idx]
    #   2. sqrsum partial for pre RMS
    #   3. GEMM partial for several mix output columns
    with T.Kernel(
        num_tokens,
        num_mix_output_tiles,
        split_k,
        threads=n_thr,
    ) as (token_idx, mix_output_tile_idx, hidden_split_idx):
        thread_idx = T.get_thread_binding()
        warp_idx = T.get_warp_idx()
        lane_idx = T.get_lane_idx()

        warp_partials = T.alloc_shared((num_warps, tile_mix_outputs + 1), T.float32)
        post_mix_smem = T.alloc_shared((hc,), T.float32)
        comb_mix_smem = T.alloc_shared((hc, hc), T.float32)

        post_mix_for_token = T.alloc_local((hc,), T.float32)
        comb_mix_for_token = T.alloc_local((hc, hc), T.float32)

        mix_acc = T.alloc_local((tile_mix_outputs,), T.float32)
        sqrsum_acc = T.alloc_local((1,), T.float32)
        cur_residual_values = T.alloc_local((hc,), T.float32)

        T.clear(mix_acc)
        T.clear(sqrsum_acc)

        hidden_split_start = hidden_split_idx * hidden_per_split

        if ENABLE_PDL:
            T.pdl_sync()

        # Load post/comb coefficients for this token.
        #
        # PyTorch equivalent:
        #   post = prev_post_mix[token_idx]      # [HC]
        #   comb = prev_comb_mix[token_idx]      # [HC, HC]
        T.copy(prev_post_mix[token_idx, 0], post_mix_smem)
        T.copy(prev_comb_mix[token_idx, 0, 0], comb_mix_smem)

        for route_idx in T.unroll(hc):
            post_mix_for_token[route_idx] = post_mix_smem[route_idx]

        for old_route_idx in T.unroll(hc):
            for new_route_idx in T.unroll(hc):
                comb_mix_for_token[old_route_idx, new_route_idx] = comb_mix_smem[
                    old_route_idx, new_route_idx
                ]

        for hidden_iter in T.serial(hidden_iters_per_thread):
            hidden_idx = hidden_split_start + hidden_iter * n_thr + thread_idx

            if hidden_idx < hidden_size:
                # Step A: fused post.
                #
                # PyTorch equivalent:
                #   cur_residual =
                #       post.unsqueeze(-1) * hidden_in.unsqueeze(1)
                #       + (
                #           comb.unsqueeze(-1)
                #           * prev_residual.unsqueeze(2)
                #         ).sum(dim=1)
                #
                # Scalar form for this token and this hidden position:
                #   cur_residual[j, h]
                #     = post[j] * hidden_in[h]
                #     + sum_k comb[k, j] * prev_residual[k, h]
                for new_route_idx in T.unroll(hc):
                    cur_residual_values[new_route_idx] = (
                        post_mix_for_token[new_route_idx]
                        * hidden_in[token_idx, hidden_idx]
                    )

                    for old_route_idx in T.unroll(hc):
                        cur_residual_values[new_route_idx] += (
                            comb_mix_for_token[old_route_idx, new_route_idx]
                            * prev_residual[token_idx, old_route_idx, hidden_idx]
                        )

                # Match the unfused path:
                #   mhc_post writes bf16 residual,
                #   then mhc_pre reads bf16 residual.
                for route_idx in T.unroll(hc):
                    cur_residual_values[route_idx] = T.bfloat16(
                        cur_residual_values[route_idx]
                    )

                # Step B1: pre sqrsum partial.
                #
                # PyTorch equivalent:
                #   x_flat = cur_residual.reshape(T, HC * H).float()
                #   sqrsum = (x_flat * x_flat).sum(dim=-1)
                #
                # Only mix_output_tile_idx == 0 writes cur_residual and sqrsum,
                # otherwise different output-column CTAs would duplicate this work.
                if mix_output_tile_idx == 0:
                    for route_idx in T.unroll(hc):
                        cur_residual_out[token_idx, route_idx, hidden_idx] = (
                            cur_residual_values[route_idx]
                        )
                        sqrsum_acc[0] += (
                            cur_residual_values[route_idx]
                            * cur_residual_values[route_idx]
                        )

                # Step B2: pre GEMM partial.
                #
                # PyTorch equivalent:
                #   mixes = F.linear(x_flat, fn)
                #
                # Scalar form:
                #   mixes[token, o] +=
                #       pre_fn[o, route, hidden] * cur_residual[route, hidden]
                #
                # This CTA computes only tile_mix_outputs columns of mixes.
                for tile_col_idx in T.unroll(tile_mix_outputs):
                    mix_output_idx = (
                        mix_output_tile_idx * tile_mix_outputs + tile_col_idx
                    )

                    if mix_output_idx < num_mix_outputs:
                        for route_idx in T.unroll(hc):
                            mix_acc[tile_col_idx] += (
                                pre_fn[mix_output_idx, route_idx, hidden_idx]
                                * cur_residual_values[route_idx]
                            )

        # Reduce thread partials inside each warp.
        for tile_col_idx in T.unroll(tile_mix_outputs):
            mix_acc[tile_col_idx] = T.warp_reduce_sum(mix_acc[tile_col_idx])

        if mix_output_tile_idx == 0:
            sqrsum_acc[0] = T.warp_reduce_sum(sqrsum_acc[0])

        # One lane per warp writes warp-level partials to shared memory.
        if lane_idx == 0:
            for tile_col_idx in T.unroll(tile_mix_outputs):
                warp_partials[warp_idx, tile_col_idx] = mix_acc[tile_col_idx]

            if mix_output_tile_idx == 0:
                warp_partials[warp_idx, tile_mix_outputs] = sqrsum_acc[0]

        T.sync_threads()

        # Reduce across warps and write split partials.
        #
        # The full PyTorch result would be:
        #   mixes = F.linear(cur_residual.reshape(T, HC * H), fn)
        #   sqrsum = (cur_residual.float() ** 2).sum(dim=(1, 2))
        #
        # This kernel is split along hidden, so each CTA writes only:
        #   mixes_partial_out[hidden_split_idx, token, o]
        #   sqrsum_partial_out[hidden_split_idx, token]
        #
        # Later mhc_pre_big_fuse does:
        #   mixes = mixes_partial_out.sum(dim=0)
        #   sqrsum = sqrsum_partial_out.sum(dim=0)
        #   rms = rsqrt(sqrsum / (HC * H) + eps)
        #   mixes *= rms
        #   mixes -> pre/post/comb
        #   layer_input = sum_j pre[j] * cur_residual[j]
        if warp_idx == 0:
            for tile_col_idx in T.unroll(tile_mix_outputs):
                mix_output_idx = mix_output_tile_idx * tile_mix_outputs + tile_col_idx

                if mix_output_idx < num_mix_outputs and lane_idx == tile_col_idx:
                    mix_output_partial = T.alloc_var(T.float32, init=0.0)

                    for reduce_warp_idx in T.unroll(num_warps):
                        mix_output_partial += warp_partials[
                            reduce_warp_idx, tile_col_idx
                        ]

                    mixes_partial_out[hidden_split_idx, token_idx, mix_output_idx] = (
                        mix_output_partial
                    )

            if mix_output_tile_idx == 0 and lane_idx == 0:
                sqrsum_partial = T.alloc_var(T.float32, init=0.0)

                for reduce_warp_idx in T.unroll(num_warps):
                    sqrsum_partial += warp_partials[reduce_warp_idx, tile_mix_outputs]

                sqrsum_partial_out[hidden_split_idx, token_idx] = sqrsum_partial

        if ENABLE_PDL:
            T.pdl_trigger()
```

## 来源与验证

- 来源：[sgl-project/sglang · mhc.py](https://github.com/sgl-project/sglang/blob/2a6c2045705bd2ac46136cece18f0d32c9c13650/python/sglang/kernels/ops/layernorm/mhc.py)，commit `2a6c2045705bd2ac46136cece18f0d32c9c13650`。
- 摘录范围：L1–L1, L2–L2, L5–L5, L6–L6, L12–L12, L124–L124, L126–L126, L129–L137, L140–L157, L160–L177, L180–L207, L210–L210, L211–L211, L1361–L1605。仅选取核心及其模块内依赖，保留函数体；去除与所选入口无关的外围代码。本文是摘录与分析，非上游原文件。
- 验证：[上游测试或参考入口](https://github.com/sgl-project/sglang/blob/2a6c2045705bd2ac46136cece18f0d32c9c13650/test/registered/kernels/ops/layernorm/test_mhc_kernels.py)；本轮只核查入口与源码，未执行 GPU 正确性、编译或性能测试。
- 代码版权与许可遵循[上游 LICENSE](../_licenses/sglang.txt)；代码中的原有版权声明保留。核查日期：2026-10-29。
