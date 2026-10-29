# SGLang：128 元素分组的 FP8 激活量化

把连续激活按 128 个元素求绝对值最大值、生成 scale 并写出 FP8；可选将 scale 向上取为 2 的幂。

## 什么时候选择

- **适用**：BF16 连续输入，最后一维可整除 128；包装层会展平成二维。block_size 必须取 128，FP8 格式与 scale 布局需匹配下游。
- **不适用**：任意 block_size、非连续视图，或把 ROCm 使用路径当成 CUDA 默认最快路径。
- **性能依据**：以下收益是源码机制分析，不代表已测得加速；没有本地 GPU 结果。

## 为什么这样设计

| 原来的问题 | 设计选择 | 收益与代价 |
| --- | --- | --- |
| 量化与求 scale 分开会重复读输入 | 在 act_quant_kernel 的同一 tile 内归约、缩放和转换 | 减少中间访存；每组范围受最大值支配 |
| 下游要求幂次 scale | fast_round_scale 用位运算构造向上取整的幂次 | 格式匹配；量化误差与普通 scale 不同 |
| 不同计算路径资源需求不同 | round_scale 分支使用不同 num_stages | 减少不合适的流水开销；不能照搬到其他 shape |

**下一步验证**：先检查全零、异常大值、非整组列宽及 scale_fmt；再按同一误差容限比较 Triton 与 TileLang 的延迟。

## 核心完整代码

入口：`act_quant`。保留完整包装函数和 scale 辅助函数。依赖 SGLang 的 FP8 格式判断；上游包装参数名不代表核心支持任意分组。

```python
from typing import Optional, Tuple

import tilelang

import tilelang.language as T

import torch

from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz

pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}

if hasattr(tilelang.PassConfigKey, "TL_DISABLE_FAST_MATH"):
    pass_configs[tilelang.PassConfigKey.TL_DISABLE_FAST_MATH] = True
elif hasattr(tilelang.PassConfigKey, "TL_ENABLE_FAST_MATH"):
    pass_configs[tilelang.PassConfigKey.TL_ENABLE_FAST_MATH] = False

_is_fp8_fnuz = is_fp8_fnuz()

BF16 = "bfloat16"

FP8 = "float8_e4m3fnuz" if _is_fp8_fnuz else "float8_e4m3fn"

FP32 = "float32"

def fast_log2_ceil(x):
    bits_x = T.reinterpret("uint32", x)
    exp_x = (bits_x >> 23) & 0xFF
    man_bits = bits_x & ((1 << 23) - 1)
    return T.Cast("int32", exp_x - 127 + T.if_then_else(man_bits != 0, 1, 0))

def fast_pow2(x):
    bits_x = (x + 127) << 23
    return T.reinterpret("float32", bits_x)

def fast_round_scale(amax, fp8_max_inv):
    return fast_pow2(fast_log2_ceil(amax * fp8_max_inv))

@tilelang.jit(pass_configs=pass_configs)
def act_quant_kernel(
    N, in_dtype=BF16, out_dtype=FP8, scale_dtype=FP32, round_scale=False
):
    M = T.symbolic("M")
    fp8_min = -224.0 if _is_fp8_fnuz else -448.0
    fp8_max = 224.0 if _is_fp8_fnuz else 448.0
    fp8_max_inv = 1 / fp8_max
    num_stages = 0 if round_scale else 2
    blk_m = 32
    group_size = 128

    @T.prim_func
    def act_quant_kernel_(
        X: T.Tensor[(M, N), in_dtype],
        Y: T.Tensor[(M, N), out_dtype],
        S: T.Tensor[(M, T.ceildiv(N, group_size)), scale_dtype],
    ):
        with T.Kernel(T.ceildiv(M, blk_m), T.ceildiv(N, group_size), threads=128) as (
            pid_m,
            pid_n,
        ):
            x_shared = T.alloc_shared((blk_m, group_size), in_dtype)
            x_local = T.alloc_fragment((blk_m, group_size), in_dtype)
            amax_local = T.alloc_fragment((blk_m,), scale_dtype)
            s_local = T.alloc_fragment((blk_m,), scale_dtype)
            y_local = T.alloc_fragment((blk_m, group_size), out_dtype)
            y_shared = T.alloc_shared((blk_m, group_size), out_dtype)

            for _ in T.Pipelined(1, num_stages=num_stages):
                T.copy(X[pid_m * blk_m, pid_n * group_size], x_shared)
                T.copy(x_shared, x_local)
                T.reduce_absmax(x_local, amax_local, dim=1)
                for i in T.Parallel(blk_m):
                    amax_local[i] = T.max(amax_local[i], 1e-4)
                    if round_scale:
                        s_local[i] = fast_round_scale(amax_local[i], fp8_max_inv)
                    else:
                        s_local[i] = amax_local[i] * fp8_max_inv
                for i, j in T.Parallel(blk_m, group_size):
                    y_local[i, j] = T.clamp(
                        x_local[i, j] / s_local[i], fp8_min, fp8_max
                    )
                for i in T.Parallel(blk_m):
                    S[pid_m * blk_m + i, pid_n] = s_local[i]
                T.copy(y_local, y_shared)
                T.copy(y_shared, Y[pid_m * blk_m, pid_n * group_size])

    return act_quant_kernel_

def act_quant(
    x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization.

    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be contiguous and its last dimension size must be divisible by `block_size`.
        block_size (int, optional): The size of the blocks to be used for quantization. Default is 128.
        scale_fmt (Optional[str], optional): The format of the scale. Default is None.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The quantized tensor with dtype `torch.float8_e4m3fn`.
            - A tensor of scaling factors with dtype `torch.float32`.
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert x.size(-1) % block_size == 0, (
        f"Last dimension size must be divisible by block_size (block_size={block_size})"
    )
    N = x.size(-1)
    if _is_fp8_fnuz:
        y = torch.empty_like(x, dtype=torch.float8_e4m3fnuz)
    else:
        y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = x.new_empty(*x.size()[:-1], N // block_size, dtype=torch.float32)
    kernel = act_quant_kernel(N, round_scale=scale_fmt is not None)
    kernel(x.view(-1, N), y.view(-1, N), s.view(-1, N // block_size))
    return y, s
```

## 来源与验证

- 来源：[sgl-project/sglang · act_quant_tilelang.py](https://github.com/sgl-project/sglang/blob/2a6c2045705bd2ac46136cece18f0d32c9c13650/python/sglang/kernels/ops/quantization/act_quant_tilelang.py)，commit `2a6c2045705bd2ac46136cece18f0d32c9c13650`。
- 摘录范围：L1–L1, L3–L3, L4–L4, L5–L5, L7–L7, L11–L14, L16–L19, L21–L21, L23–L23, L24–L24, L25–L25, L28–L32, L35–L37, L40–L41, L44–L92, L95–L122。仅选取核心及其模块内依赖，保留函数体；去除与所选入口无关的外围代码。本文是摘录与分析，非上游原文件。
- 验证：[上游测试或参考入口](https://github.com/sgl-project/sglang/blob/2a6c2045705bd2ac46136cece18f0d32c9c13650/test/manual/kernels/quantization/test_act_quant_triton.py)；本轮只核查入口与源码，未执行 GPU 正确性、编译或性能测试。
- 代码版权与许可遵循[上游 LICENSE](../_licenses/sglang.txt)；代码中的原有版权声明保留。核查日期：2026-10-29。
