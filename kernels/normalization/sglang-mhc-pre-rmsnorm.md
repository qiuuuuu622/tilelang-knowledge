# SGLang mHC：pre mapping 与 RMSNorm 融合

消费 GEMM 的分段结果，生成 mHC mixing 系数和加权层输入，再在片上完成 RMSNorm。

## 什么时候选择

- **适用**：mHC 的 hc_mult、Sinkhorn 次数、BF16 residual 与 FP32 partial 布局一致；前置 GEMM 已完成。
- **不适用**：当作包含前置投影 GEMM 的完整 mHC pre；只需普通 RMSNorm 的任务。
- **性能依据**：以下收益是源码机制分析，不代表已测得加速；没有本地 GPU 结果。

## 为什么这样设计

| 原来的问题 | 设计选择 | 收益与代价 |
| --- | --- | --- |
| mixing 需要 split GEMM 结果 | 同一 CTA 汇总平方和与投影 partial | 减少额外归约 launch；仍需前置 GEMM |
| 组合矩阵需要归一化 | sigmoid 与 Sinkhorn 跟随 mixing 计算 | 缩短中间张量生命周期；迭代次数影响开销 |
| 层输入写回后又被 RMSNorm 读取 | 加权结果放共享内存，两次片上扫描 | 减少一次 HBM 往返；共享内存随 hidden 增长 |

**下一步验证**：对照未融合 mHC pre + RMSNorm，检查 BF16 舍入位置及 Sinkhorn 误差；测共享内存占用和整段延迟。

## 核心完整代码

入口：`mhc_pre_big_fuse_with_norm_tilelang`。核心工厂及原有 lazy TileLang 初始化完整保留；仍依赖 SGLang 的 PDL 能力判断。

```python
import functools

import importlib

import math

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
):
    """Fused mhc_pre big_fuse + RMSNorm of layer_input.

    Identical to mhc_pre_big_fuse_tilelang for the (post_mix, comb_mix) path.
    For the layer_input path, the weighted-sum result is stashed in shared
    memory while accumulating sum_sq, then a second pipelined sweep applies
    rsqrt(sum_sq/D + norm_eps) * norm_weight before writing to HBM.
    """
    num_tokens = T.dynamic("num_tokens")
    hc_mult3 = hc_mult * (2 + hc_mult)
    if gemm_last_dim < 0:
        gemm_last_dim = hc_mult3
    hidden_block = math.gcd(1024, hidden_size)

    gemm_out_mul: T.Tensor[[n_splits, num_tokens, gemm_last_dim], T.float32]
    gemm_out_sqrsum: T.Tensor[[n_splits, num_tokens], T.float32]
    hc_scale: T.Tensor[[3], T.float32]
    hc_base: T.Tensor[[hc_mult3], T.float32]
    residual: T.Tensor[[num_tokens, hc_mult, hidden_size], T.bfloat16]
    post_mix: T.Tensor[[num_tokens, hc_mult], T.float32]
    comb_mix: T.Tensor[[num_tokens, hc_mult * hc_mult], T.float32]
    layer_input: T.Tensor[[num_tokens, hidden_size], T.bfloat16]
    norm_weight: T.Tensor[[hidden_size], T.bfloat16]

    ENABLE_PDL = is_arch_support_pdl()
    with T.Kernel(num_tokens, threads=96) as i:
        rms = T.alloc_fragment(1, T.float32)
        mixes = T.alloc_fragment(hc_mult3, T.float32)
        T.clear(mixes)
        rms[0] = 0

        if ENABLE_PDL:
            T.pdl_sync()

        for i_split in T.serial(n_splits):
            rms[0] += gemm_out_sqrsum[i_split, i]
        rms[0] = T.rsqrt(rms[0] / (hc_mult * hidden_size) + rms_eps)
        for j in T.Parallel(hc_mult3):
            mixes[j] = 0
            for i_split in T.serial(n_splits):
                mixes[j] += gemm_out_mul[i_split, i, j]
            mixes[j] *= rms[0]
        mixes_shared = T.alloc_shared(hc_mult3, T.float32)
        T.copy(mixes, mixes_shared)

        if T.get_thread_binding() < 32:
            cm = T.alloc_fragment((hc_mult, hc_mult), T.float32)
            for j in T.Parallel(hc_mult):
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

            for _ in T.serial(sinkhorn_repeat - 1):
                T.reduce_sum(cm, row_sum, dim=1)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / (row_sum[j] + hc_sinkhorn_eps)

                T.reduce_sum(cm, col_sum, dim=0)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

            for j, k in T.Parallel(hc_mult, hc_mult):
                comb_mix[i, j * hc_mult + k] = cm[j, k]
        else:
            pre_mix_shared = T.alloc_shared(hc_mult, T.float32)
            for j in T.Parallel(hc_mult):
                pre_mix_shared[j] = (
                    T.sigmoid(
                        mixes_shared[j] * hc_scale[0] + hc_base[j],
                    )
                    + hc_pre_eps
                )

            # Stash unnormalized weighted-sum output in shared memory as bf16
            # (matches the rounding the reference path does when RMSNorm reads bf16).
            output_shared = T.alloc_shared(hidden_size, T.bfloat16)
            sumsq_per_pos = T.alloc_fragment(hidden_block, T.float32)
            T.clear(sumsq_per_pos)

            for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=3):
                xs = T.alloc_shared((hc_mult, hidden_block), T.bfloat16)
                xl = T.alloc_fragment((hc_mult, hidden_block), T.float32)
                T.copy(residual[i, 0, i0_h * hidden_block], xs)
                T.copy(xs, xl)

                ol = T.alloc_fragment(hidden_block, T.float32)
                T.clear(ol)

                for i_hc in T.serial(hc_mult):
                    pre = pre_mix_shared[i_hc]
                    for i1_h in T.Parallel(hidden_block):
                        ol[i1_h] += pre * xl[i_hc, i1_h]

                for i1_h in T.Parallel(hidden_block):
                    sumsq_per_pos[i1_h] += ol[i1_h] * ol[i1_h]
                    output_shared[i0_h * hidden_block + i1_h] = T.bfloat16(ol[i1_h])

            sumsq = T.alloc_fragment(1, T.float32)
            T.reduce_sum(sumsq_per_pos, sumsq, dim=0)
            rsqrt_norm = T.alloc_fragment(1, T.float32)
            rsqrt_norm[0] = T.rsqrt(sumsq[0] / hidden_size + norm_eps)

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

- 来源：[sgl-project/sglang · mhc.py](https://github.com/sgl-project/sglang/blob/2a6c2045705bd2ac46136cece18f0d32c9c13650/python/sglang/kernels/ops/layernorm/mhc.py)，commit `2a6c2045705bd2ac46136cece18f0d32c9c13650`。
- 摘录范围：L1–L1, L2–L2, L4–L4, L5–L5, L6–L6, L12–L12, L124–L124, L126–L126, L129–L137, L140–L157, L160–L177, L180–L207, L210–L210, L211–L211, L890–L1059。仅选取核心及其模块内依赖，保留函数体；去除与所选入口无关的外围代码。本文是摘录与分析，非上游原文件。
- 验证：[上游测试或参考入口](https://github.com/sgl-project/sglang/blob/2a6c2045705bd2ac46136cece18f0d32c9c13650/test/registered/kernels/ops/layernorm/test_mhc_kernels.py)；本轮只核查入口与源码，未执行 GPU 正确性、编译或性能测试。
- 代码版权与许可遵循[上游 LICENSE](../_licenses/sglang.txt)；代码中的原有版权声明保留。核查日期：2026-10-29。
