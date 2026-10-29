# TileKernels：可配置的 per-token 量化

将分组 amax、scale 和类型转换组织为同一工厂，支持输入已有 scale 与可选随机舍入。

## 什么时候选择

- **适用**：使用 CastInputConfig/CastOutputConfig 定义的类型、scale 粒度及存储布局；向量宽度和分组满足源码断言。
- **不适用**：将 per-token 一概理解为整行一个 scale；忽略 FP4 打包、scale stride 或输入反量化。
- **性能依据**：以下收益是源码机制分析，不代表已测得加速；没有本地 GPU 结果。

## 为什么这样设计

| 原来的问题 | 设计选择 | 收益与代价 |
| --- | --- | --- |
| 量化转换有多种配置 | 配置在编译期决定 sf_only/cast_only/完整路径 | 消除不需要的工作；变体和契约增加 |
| 小 batch 与普通 batch 工作量不同 | small_batch 改线程数及每线程元素 | 调节并行度；不能默认更大线程数更快 |
| 反量化再量化会多次访存 | 在同一 tile 消费输入 scale 并写输出 | 减少中间张量；舍入语义需单独验证 |

**下一步验证**：按 dtype、sf_block、stride 验证反量化误差；随机舍入需统计误差分布，随后测 small_batch 分界与有效带宽。

## 核心完整代码

入口：`get_per_token_cast_kernel_cuda`。完整工厂与 rbits_hash；Cast 配置、scale/打包辅助函数由 tile_kernels.quant.common 提供，必须使用同版本工程。

```python
import math

import tilelang

from tilelang import language as T

from tile_kernels.utils import align, is_power_of_two

from tile_kernels.quant.common import *

@T.macro
def rbits_hash(values, quad_idx):
    h = T.alloc_var(T.uint32)
    h = (
        T.reinterpret(values[0], T.uint32) * T.uint32(0x5671D42B)
        + T.reinterpret(values[1], T.uint32) * T.uint32(0x9995E499)
        + T.reinterpret(values[2], T.uint32) * T.uint32(0xACE1B8A5)
        + T.reinterpret(values[3], T.uint32) * T.uint32(0xE153538D)
        + T.cast(quad_idx, T.uint32)
    )
    h = h ^ (h >> 23)
    h = h * T.uint32(0x7FEB352D)
    h = h ^ (h >> 16)
    h = h * T.uint32(0x846CA68B)
    h = h ^ (h >> 11)
    return h

@tilelang.jit(
    target='cuda',
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_LOWER_LDGSTG_PREDICATED: True,
    },
)
def get_per_token_cast_kernel_cuda(
    hidden: int,
    token_stride: int,
    in_config: CastInputConfig,
    out_config: CastOutputConfig,
    sf_only: bool = False,
    cast_only: bool = False,
    small_batch: bool = False,
    stochastic_cast: bool = False,
    use_pdl: bool = False,
):
    if small_batch:
        num_threads, num_elems_per_thread = 64, 32
    else:
        num_threads, num_elems_per_thread = 32, 64

    num_elems_per_block = num_threads * num_elems_per_thread
    num_per_channels = out_config.sf_block[1]

    if is_power_of_two(num_per_channels):
        block_k = max(num_per_channels, math.gcd(num_elems_per_block, hidden))
    elif hidden == num_per_channels:
        assert not in_config.with_sf and not cast_only and not sf_only
        # Must ensure each thread has elements as multiple of 2 for FP4
        block_k = align(hidden, num_threads * (2 if out_config.dtype == T.float4_e2m1fn else 1))
        num_per_channels = block_k
    else:
        raise ValueError('Cast granularity must be equal to hidden or be power of two')

    assert block_k % num_per_channels == 0
    block_m = 1 if num_elems_per_block % block_k != 0 else num_elems_per_block // block_k
    num_groups = block_k // num_per_channels

    num_vectorize = min(get_best_vectorize_size(in_config.dtype), math.gcd(block_m * block_k // num_threads, 32))
    sf_value_dtype = T.float32 if in_config.with_sf else in_config.dtype
    sf_inv_dtype = get_sf_inv_dtype(sf_value_dtype, out_config)
    if stochastic_cast:
        assert num_vectorize >= 4, f'stochastic_cast requires vectorize width >= 4, got {num_vectorize}'

    if in_config.with_sf:
        assert not cast_only and not sf_only
        assert block_k % num_vectorize == 0
        assert num_per_channels >= num_vectorize, (
            f'num_per_channels ({num_per_channels}) must be >= num_vectorize ({num_vectorize}); on sm>=10 with float8, num_vectorize=32, so num_per_channels=16 would cause a zero-dimension reshape'
        )
        assert block_m % in_config.sf_block[0] == 0 or in_config.sf_block[0] % block_m == 0
        assert block_k % in_config.sf_block[1] == 0 or in_config.sf_block[1] % block_k == 0

    # Runtime symbols
    num_tokens = T.dynamic('num_tokens')
    in_sf_stride = T.dynamic('in_sf_stride')
    out_sf_stride = T.dynamic('out_sf_stride')
    x_sf_shape = get_sf_shape((num_tokens, hidden), in_config)
    sf_shape = get_sf_shape((num_tokens, hidden), out_config)

    def x_layout_fn(i: int, j: int):
        id = i * block_k + j
        return id // num_vectorize % num_threads, id // (num_vectorize * num_threads) * num_vectorize + id % num_vectorize

    blocks_per_token = T.ceildiv(hidden, block_k)

    @T.prim_func
    def per_token_cast_kernel(
        x: T.StridedTensor[(num_tokens, hidden), (token_stride, 1), in_config.dtype],
        x_sf: T.StridedTensor[x_sf_shape, (in_sf_stride, 1), in_config.sf_dtype],
        out: T.Tensor[(num_tokens, hidden), out_config.dtype],
        out_sf: T.StridedTensor[sf_shape, (out_sf_stride, 1), out_config.sf_dtype],
    ):
        T.assume(num_tokens > 0)
        with T.Kernel(T.ceildiv(num_tokens, block_m) * blocks_per_token, threads=num_threads) as pid:
            pid_token = pid // blocks_per_token
            pid_hidden = pid % blocks_per_token
            if use_pdl:
                T.pdl_sync()
            x_fragment = T.alloc_fragment((block_m, block_k), in_config.dtype)
            sf_inv_fragment = T.alloc_fragment((block_m, num_groups), sf_inv_dtype)
            if stochastic_cast or in_config.with_sf:
                out_fragment_or_shared = T.alloc_shared((block_m, block_k), out_config.dtype)
            else:
                out_fragment_or_shared = T.alloc_fragment((block_m, block_k), out_config.dtype)

            T.annotate_layout({
                x_fragment: T.Fragment(
                    (block_m, block_k),
                    forward_fn=x_layout_fn,
                )
            })  # fmt: off

            # Copy input into registers
            T.copy(x[pid_token * block_m, pid_hidden * block_k], x_fragment, disable_tma=True)

            if in_config.with_sf:
                num_sf_rows_per_block = T.ceildiv(block_m, in_config.sf_block[0])
                num_sf_cols_per_block = T.ceildiv(block_k, in_config.sf_block[1])
                x_sf_fragment = T.alloc_fragment((num_sf_rows_per_block, num_sf_cols_per_block), T.float32)
                for i, j in T.Parallel(num_sf_rows_per_block, num_sf_cols_per_block):
                    m_idx = pid_token * block_m // in_config.sf_block[0] + i
                    k_idx = pid_hidden * block_k // in_config.sf_block[1] + j
                    x_sf_fragment[i, j] = transform_sf(load_sf(x_sf, m_idx, k_idx, in_config), in_config)

                # Reduce stage 1, use half for reduction
                stage1_amax_fragment = T.alloc_fragment((block_m, block_k // num_vectorize), T.float16)
                x_stage1_fragment_reshaped = T.reshape(x_fragment, [block_m, block_k // num_vectorize, num_vectorize])
                T.reduce_absmax(x_stage1_fragment_reshaped, stage1_amax_fragment, dim=-1)

                # Apply scaling factor
                stage2_amax_fragment = T.alloc_fragment((block_m, block_k // num_vectorize), T.float32)
                T.clear(stage2_amax_fragment)
                for i, j in T.Parallel(block_m, block_k // num_vectorize):
                    stage2_amax_fragment[i, j] = (
                        T.float32(stage1_amax_fragment[i, j]) * x_sf_fragment[i // in_config.sf_block[0], j * num_vectorize // in_config.sf_block[1]]
                    )

                # Reduce stage 2, using float for reduction
                stage2_amax_fragment_reshaped = T.reshape(stage2_amax_fragment, [block_m, num_groups, block_k // num_vectorize // num_groups])
                T.reduce_max(stage2_amax_fragment_reshaped, sf_inv_fragment, dim=-1)

                for i, j in T.Parallel(block_m, num_groups):
                    sf, sf_inv = get_sf_and_inv(sf_inv_fragment[i, j], out_config, sf_inv_dtype)
                    # Store SF
                    m_idx = pid_token * block_m + i
                    k_idx = pid_hidden * num_groups + j
                    store_sf(out_sf, sf, m_idx, k_idx, out_config)
                    sf_inv_fragment[i, j] = sf_inv

                # Store casted values
                if not sf_only:
                    if stochastic_cast:
                        x_fp32_local = T.alloc_local((4,), T.float32)
                        for i, j in T.Parallel(block_m, block_k // 4):
                            # Apply two multiplication at once to save the number of multiplication calculated
                            for k in T.vectorized(4):
                                factor = (
                                    x_sf_fragment[i // in_config.sf_block[0], (j * 4 + k) // in_config.sf_block[1]]
                                    * sf_inv_fragment[i, (j * 4 + k) // num_per_channels]
                                )
                                x_fp32_local[k] = T.float32(x_fragment[i, j * 4 + k]) * factor
                            rbits = rbits_hash(x_fp32_local, pid_hidden * (block_k // 4) + j)
                            for k in T.vectorized(4):
                                out_fragment_or_shared[i, j * 4 + k] = T.cast(x_fp32_local[k], out_config.dtype, round='rs', rbits=rbits)
                    else:
                        for i, j in T.Parallel(block_m, block_k):
                            # Apply two multiplication at once to save the number of multiplication calculated
                            factor = x_sf_fragment[i // in_config.sf_block[0], j // in_config.sf_block[1]] * sf_inv_fragment[i, j // num_per_channels]
                            out_fragment_or_shared[i, j] = T.float32(x_fragment[i, j]) * factor

            else:
                if cast_only:
                    for i, j in T.Parallel(block_m, num_groups):
                        sf = transform_sf(load_sf(out_sf, pid_token * block_m + i, pid_hidden * num_groups + j, out_config), out_config)
                        sf_inv_fragment[i, j] = 1 / sf
                else:
                    amax_fragment = T.alloc_fragment((block_m, num_groups), in_config.dtype)
                    x_fragment_reshaped = T.reshape(x_fragment, [block_m, num_groups, num_per_channels])
                    # Reduce SF
                    T.reduce_absmax(x_fragment_reshaped, amax_fragment, dim=2)
                    for i, j in T.Parallel(block_m, num_groups):
                        amax = T.cast(amax_fragment[i, j], T.float32)
                        sf, sf_inv = get_sf_and_inv(amax, out_config, sf_inv_dtype)

                        # Store SF
                        m_idx = pid_token * block_m + i
                        k_idx = pid_hidden * num_groups + j
                        store_sf(out_sf, sf, m_idx, k_idx, out_config)
                        sf_inv_fragment[i, j] = sf_inv

                # Store casted values
                if not sf_only:
                    if stochastic_cast:
                        x_fp32_local = T.alloc_local((4,), T.float32)
                        for i, j in T.Parallel(block_m, block_k // 4):
                            for k in T.vectorized(4):
                                x_fp32_local[k] = x_fragment[i, j * 4 + k] * sf_inv_fragment[i, (j * 4 + k) // num_per_channels]
                            rbits = rbits_hash(x_fp32_local, pid_hidden * (block_k // 4) + j)
                            for k in T.vectorized(4):
                                out_fragment_or_shared[i, j * 4 + k] = T.cast(x_fp32_local[k], out_config.dtype, round='rs', rbits=rbits)
                    else:
                        for i, j in T.Parallel(block_m, block_k):
                            out_fragment_or_shared[i, j] = x_fragment[i, j] * sf_inv_fragment[i, j // num_per_channels]

            if not sf_only:
                T.copy(out_fragment_or_shared, out[pid_token * block_m, pid_hidden * block_k], disable_tma=True)

            if use_pdl:
                T.pdl_trigger()

    return per_token_cast_kernel
```

## 来源与验证

- 来源：[deepseek-ai/TileKernels · per_token_cast_cuda.py](https://github.com/deepseek-ai/TileKernels/blob/c1cf914a82e6901c679445f872051ce44ac2cc63/tile_kernels/quant/per_token_cast_cuda.py)，commit `c1cf914a82e6901c679445f872051ce44ac2cc63`。
- 摘录范围：L1–L1, L3–L3, L4–L4, L6–L6, L7–L7, L10–L25, L28–L221。仅选取核心及其模块内依赖，保留函数体；去除与所选入口无关的外围代码。本文是摘录与分析，非上游原文件。
- 验证：[上游测试或参考入口](https://github.com/deepseek-ai/TileKernels/blob/c1cf914a82e6901c679445f872051ce44ac2cc63/tests/quant/test_per_token_cast.py)；本轮只核查入口与源码，未执行 GPU 正确性、编译或性能测试。
- 代码版权与许可遵循[上游 LICENSE](../_licenses/tilekernels.txt)；代码中的原有版权声明保留。核查日期：2026-10-29。
