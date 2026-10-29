# TileKernels：二维 block scale 量化

在二维 token×channel 块内归约 amax 并量化，与 per-token 不同的是 scale 跨多个 token 共享。

## 什么时候选择

- **适用**：scale block 两维均取源码支持的 32 或 128，hidden 与向量/归约布局兼容。
- **不适用**：后续算子需要逐 token scale；不同输入量化契约未经检查就接入。
- **性能依据**：以下收益是源码机制分析，不代表已测得加速；没有本地 GPU 结果。

## 为什么这样设计

| 原来的问题 | 设计选择 | 收益与代价 |
| --- | --- | --- |
| 二维分组归约容易重复交换数据 | amax_forward_fn 显式组织归约 fragment | 匹配线程持有的数据；布局依赖 block 参数 |
| scale 与量化分开读取输入 | transform_fragment 同时生成 scale 与转换 | 减少重复访存；受共享分组数影响 |
| 尾部判断影响热路径 | 区分完整 tile 与边界处理 | 改善完整 tile 代码；尾部必须单独测试 |

**下一步验证**：覆盖 token/hidden 尾块、全零与离群值，比较 block scale 的精度代价，再测分组大小对字节带宽的影响。

## 核心完整代码

入口：`get_per_block_cast_kernel_cuda`。完整工厂、嵌套宏和布局函数；依赖同版本 quant.common，保留原 pass 配置，关闭检查不代表已证明无竞争。

```python
import tilelang

from tilelang import language as T

from tile_kernels.utils import ceil_div

from tile_kernels.quant.common import *

@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    },
)
def get_per_block_cast_kernel_cuda(
    hidden: int,
    in_config: CastInputConfig,
    out_config: CastOutputConfig,
    sf_only: bool = False,
    cast_only: bool = False,
    use_pdl: bool = False,
):
    num_threads = 256
    num_elements_per_block = 8192

    num_vectorize = get_best_vectorize_size(in_config.dtype)

    num_per_tokens, num_per_channels = out_config.sf_block
    assert num_per_tokens in (32, 128) and num_per_channels in (32, 128)

    block_k = max(num_per_channels, num_elements_per_block // num_per_tokens)
    block_m = num_per_tokens
    assert block_m % num_per_tokens == 0
    assert block_k % num_per_channels == 0

    num_repeat_rows = num_threads * num_vectorize // block_k
    assert num_per_tokens % num_repeat_rows == 0, 'scaling factor block size too small to be multiple of the row repeat pattern'
    num_sf_rows_per_block = block_m // num_per_tokens
    num_sf_cols_per_block = block_k // num_per_channels
    num_sf_per_block = num_sf_rows_per_block * num_sf_cols_per_block
    sf_inv_dtype = get_sf_inv_dtype(in_config.dtype, out_config)

    # Runtime symbols
    num_tokens = T.dynamic('num_tokens')
    sf_shape = get_sf_shape((num_tokens, hidden), out_config)
    sf_stride = T.dynamic('sf_stride')

    def amax_forward_fn(i: int, j: int, k: int):
        num_threads_per_token = num_per_channels // num_vectorize
        return j * num_threads_per_token + k // num_threads_per_token * (block_k // num_vectorize) + k % num_threads_per_token, i

    @T.macro
    def transform_fragment(
        x_fragment: T.Fragment,
        out_sf: T.Tensor,
        pid_x: int,
        pid_y: int,
        sf_inv_fragment: T.Fragment,
    ):
        if not cast_only:
            amax_reducer = T.alloc_fragment((num_sf_rows_per_block, num_sf_cols_per_block, num_threads // num_sf_per_block), in_config.dtype)
            amax_fragment = T.alloc_fragment((num_sf_rows_per_block, num_sf_cols_per_block), T.float32)
            T.annotate_layout(
                {
                    amax_reducer: T.Fragment(
                        (num_sf_rows_per_block, num_sf_cols_per_block, num_threads // num_sf_per_block),
                        replicate=1,
                        forward_fn=amax_forward_fn,
                    ),
                    amax_fragment: T.Fragment(
                        (num_sf_rows_per_block, num_sf_cols_per_block),
                        replicate=num_threads // num_sf_per_block,
                        forward_fn=amax_forward_fn,
                    ),
                }
            )
            T.clear(amax_reducer)
            for i, j in T.Parallel(block_m, block_k):
                block_offset = (i % (num_threads * num_vectorize // block_k) * num_per_channels + j % num_per_channels) // num_vectorize
                amax_reducer[i // num_per_tokens, j // num_per_channels, block_offset] = T.max(
                    amax_reducer[i // num_per_tokens, j // num_per_channels, block_offset],
                    T.abs(x_fragment[i, j]),
                )
            T.reduce_max(amax_reducer, amax_fragment, dim=2)
            for i, j in T.Parallel(num_sf_rows_per_block, num_sf_cols_per_block):
                sf, sf_inv = get_sf_and_inv(amax_fragment[i, j], out_config, sf_inv_dtype)
                store_sf(out_sf, sf, pid_x * num_sf_rows_per_block + i, pid_y * num_sf_cols_per_block + j, out_config)
                sf_inv_fragment[i, j] = sf_inv
        else:
            for i, j in T.Parallel(num_sf_rows_per_block, num_sf_cols_per_block):
                sf = load_sf(out_sf, pid_x * num_sf_rows_per_block + i, pid_y * num_sf_cols_per_block + j, out_config)
                sf_inv_fragment[i, j] = 1 / sf

    @T.prim_func
    def per_block_cast_kernel(
        x: T.Tensor[(num_tokens, hidden), in_config.dtype],
        out: T.Tensor[(num_tokens, hidden), out_config.dtype],
        out_sf: T.StridedTensor[sf_shape, (sf_stride, 1), out_config.sf_dtype],
    ):
        with T.Kernel(ceil_div(num_tokens, block_m), ceil_div(hidden, block_k), threads=num_threads) as (pid_x, pid_y):
            if use_pdl:
                T.pdl_sync()
            x_fragment = T.alloc_fragment((block_m, block_k), in_config.dtype)
            sf_inv_fragment = T.alloc_fragment((num_sf_rows_per_block, num_sf_cols_per_block), sf_inv_dtype)

            T.annotate_layout(
                {
                    x_fragment: T.Fragment(
                        (block_m, block_k),
                        forward_fn=lambda i, j: (
                            (i * block_k + j) // num_vectorize % num_threads,
                            i * block_k // (num_threads * num_vectorize) * num_vectorize + j % num_vectorize,
                        ),
                    ),
                    sf_inv_fragment: T.Fragment(
                        (num_sf_rows_per_block, num_sf_cols_per_block),
                        replicate=num_threads // num_sf_per_block,
                        forward_fn=amax_forward_fn,
                    ),
                }
            )

            # Move the `if` statements to the top level to improve SASS code generation.
            if pid_x < ceil_div(num_tokens, block_m) - 1 and pid_y < ceil_div(hidden, block_k) - 1:
                T.copy(x[pid_x * block_m, pid_y * block_k], x_fragment)
                transform_fragment(x_fragment, out_sf, pid_x, pid_y, sf_inv_fragment)

                if not sf_only:
                    for i, j in T.Parallel(block_m, block_k):
                        out[pid_x * block_m + i, pid_y * block_k + j] = x_fragment[i, j] * sf_inv_fragment[i // num_per_tokens, j // num_per_channels]
            else:
                T.copy(x[pid_x * block_m, pid_y * block_k], x_fragment)
                transform_fragment(x_fragment, out_sf, pid_x, pid_y, sf_inv_fragment)

                if not sf_only:
                    for i, j in T.Parallel(block_m, block_k):
                        out[pid_x * block_m + i, pid_y * block_k + j] = x_fragment[i, j] * sf_inv_fragment[i // num_per_tokens, j // num_per_channels]

            if use_pdl:
                T.pdl_trigger()

    return per_block_cast_kernel
```

## 来源与验证

- 来源：[deepseek-ai/TileKernels · per_block_cast_cuda.py](https://github.com/deepseek-ai/TileKernels/blob/c1cf914a82e6901c679445f872051ce44ac2cc63/tile_kernels/quant/per_block_cast_cuda.py)，commit `c1cf914a82e6901c679445f872051ce44ac2cc63`。
- 摘录范围：L1–L1, L2–L2, L4–L4, L5–L5, L8–L141。仅选取核心及其模块内依赖，保留函数体；去除与所选入口无关的外围代码。本文是摘录与分析，非上游原文件。
- 验证：[上游测试或参考入口](https://github.com/deepseek-ai/TileKernels/blob/c1cf914a82e6901c679445f872051ce44ac2cc63/tests/quant/test_per_block_cast.py)；本轮只核查入口与源码，未执行 GPU 正确性、编译或性能测试。
- 代码版权与许可遵循[上游 LICENSE](../_licenses/tilekernels.txt)；代码中的原有版权声明保留。核查日期：2026-10-29。
