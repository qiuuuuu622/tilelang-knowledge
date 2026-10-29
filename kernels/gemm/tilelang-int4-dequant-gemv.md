# TileLang 官方：面向小 M 的 INT4 GEMV

把输出列分工、K 维线程归约和低比特解包融合，展示小 M 时不用常规大 GEMM tile 的选择。

## 什么时候选择

- **适用**：小 M，A 不转置、B 逻辑转置；K 和 N 满足手工向量读取与分块边界，低比特布局与 helper 一致。
- **不适用**：把 with_scaling 参数视为已完整支持外部 scale 张量；未经严格检查的尾部、量化格式或精度要求。
- **性能依据**：以下收益是源码机制分析，不代表已测得加速；没有本地 GPU 结果。

## 为什么这样设计

| 原来的问题 | 设计选择 | 收益与代价 |
| --- | --- | --- |
| 小 M 不能填满大矩阵 tile | n_partition 分输出列，reduce_thread 分 K | 增加有效线程工作；归约带来通信 |
| 权重读取占比高 | 压缩读取并在寄存器展开 | 减少字节数；解码增加指令 |
| 常规移位解码可能昂贵 | fast_decoding 可导入 LOP3 helper | 提供比较路径；依赖额外实现与格式匹配 |

**下一步验证**：必须先补严格正确性检查：上游演示使用 atol=1e3、rtol=1e-1，不能据此认定精度合格。再测普通/快速解码的总时间。

## 核心完整代码

入口：`dequantize_gemv`。完整 dequantize_gemv；依赖同目录 quantize。此条为设计参考候选，宽松的上游检查不足以支持生产可用结论。

```python
import tilelang

from tilelang import language as T

from typing import Optional, Callable, Any

from tilelang import DataType

from quantize import (
    _tir_packed_int_to_int_convert,
)

@tilelang.jit
def dequantize_gemv(
    M: int,
    N: int,
    K: int,
    in_dtype: str,
    out_dtype: str,
    accum_dtype: str,
    num_bits: int = 4,
    storage_dtype: T.dtype = T.int8,
    source_format: str = "uint",
    n_partition: int = 4,
    reduce_thread: int = 32,
    fast_decoding: bool = False,
    trans_A: bool = False,
    trans_B: bool = True,
    group_size: int = -1,
    with_scaling: bool = False,
) -> Callable[..., Any]:
    assert n_partition is not None, "n_partition must be provided"
    assert reduce_thread is not None, (
        "reduce_thread must be provided currently, as related bitblas.gpu.gemv.GEMVsch_outer_reduction_with_config is not implemented"
    )

    assert trans_A is False, "Dequantize only implement for trans_A=False currently"
    assert trans_B is True, "Dequantize only implement for trans_B=TRue currently"
    storage_type = "".join(c for c in storage_dtype if not c.isdigit())
    storage_nbit = int("".join(c for c in storage_dtype if c.isdigit()))
    num_elems_per_byte = storage_nbit // num_bits

    MAX_TRANSACTION_SIZE_IN_BITS = 128
    micro_size_k = MAX_TRANSACTION_SIZE_IN_BITS // DataType(in_dtype).bits
    micro_size_k_compressed = micro_size_k // num_elems_per_byte
    block_K = reduce_thread * micro_size_k

    if group_size == -1:
        group_size = K

    A_shape = (M, K)
    B_shape = (N, K // storage_nbit * num_bits)
    C_shape = (M, N)

    dp4a_size = 4
    use_dp4a = in_dtype == T.int8 and accum_dtype == T.int32

    import_source: Optional[str] = None
    func_name: str = ""
    if fast_decoding is True:
        # Lazy import to decrease the startup time
        # as intrin registry may take a while to load
        from quantize import get_lop3_intrin_group

        lop3_intrin_info = get_lop3_intrin_group(
            out_dtype=in_dtype,
            source_format=source_format,
            source_bit=num_bits,
            storage_dtype=storage_dtype,
            with_scaling=with_scaling,
            with_zeros=False,
        )
        import_source = lop3_intrin_info["c_source"]
        func_name = lop3_intrin_info["func_name"]
        assert import_source is not None, "lop3_intrin_info is not found"
        assert func_name is not None, "lop3_intrin_info is not found"
        import_source = import_source

    @T.prim_func
    def main(
        A: T.Tensor[A_shape, in_dtype],
        B: T.Tensor[B_shape, storage_dtype],
        C: T.Tensor[C_shape, out_dtype],
    ):
        with T.Kernel(
            T.ceildiv(N, n_partition),
            M,
            threads=(reduce_thread, n_partition),
        ) as (
            bx,
            by,
        ):
            A_local = T.alloc_local((micro_size_k,), in_dtype)
            B_quant_local = T.alloc_local([micro_size_k_compressed], storage_dtype)
            B_dequantize_local = T.alloc_local([micro_size_k], in_dtype)
            accum_res = T.alloc_local((1,), accum_dtype)
            reduced_accum_res = T.alloc_local((1,), accum_dtype)

            kr = T.thread_binding(0, reduce_thread, thread="threadIdx.x")
            ni = T.thread_binding(0, n_partition, thread="threadIdx.y")

            T.import_source(import_source)

            T.clear(accum_res)
            for ko in T.serial(T.ceildiv(K, block_K)):
                for v in T.vectorized(micro_size_k):
                    A_local[v] = A[by, ko * block_K + kr * micro_size_k + v]

                for v in T.vectorized(micro_size_k_compressed):
                    B_quant_local[v] = B[
                        bx * n_partition + ni,
                        ko * (reduce_thread * micro_size_k_compressed) + kr * micro_size_k_compressed + v,
                    ]

                if fast_decoding:
                    T.call_extern(
                        func_name,
                        T.access_ptr(B_quant_local, "r"),
                        T.access_ptr(B_dequantize_local, "w"),
                        dtype=in_dtype,
                    )
                else:
                    for ki in T.serial(micro_size_k):
                        B_dequantize_local[ki] = _tir_packed_int_to_int_convert(storage_type, storage_nbit)(
                            num_bits, B_quant_local[ki // num_elems_per_byte], ki % num_elems_per_byte, in_dtype
                        )

                if use_dp4a:
                    for ki in T.serial(micro_size_k // dp4a_size):
                        T.dp4a(
                            A_local[ki * dp4a_size],
                            B_dequantize_local[ki * dp4a_size],
                            accum_res[0],
                        )
                else:
                    for ki in T.serial(micro_size_k):
                        accum_res[0] += A_local[ki] * B_dequantize_local[ki]

            with T.attr(
                T.comm_reducer(lambda x, y: x + y, [T.cast(0, accum_dtype)]),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        accum_res[0],
                        True,
                        reduced_accum_res[0],
                        kr,
                        dtype="handle",
                    )
                )
            if kr == 0:
                C[by, bx * n_partition + ni] = reduced_accum_res[0]

    return main
```

## 来源与验证

- 来源：[tile-ai/tilelang · example_dequant_gemv_fp16xint4.py](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/examples/dequantize_gemm/example_dequant_gemv_fp16xint4.py)，commit `0a8c6716513bcdf8a728735c531c210c33077784`。
- 摘录范围：L1–L1, L2–L2, L3–L3, L5–L5, L6–L8, L11–L155。仅选取核心及其模块内依赖，保留函数体；去除与所选入口无关的外围代码。本文是摘录与分析，非上游原文件。
- 验证：[同文件的参考/检查或评测代码](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/examples/dequantize_gemm/example_dequant_gemv_fp16xint4.py)；示例测量脚本不是本仓库实测结果，本文未执行。
- 代码版权与许可遵循[上游 LICENSE](../_licenses/tilelang.txt)；代码中的原有版权声明保留。核查日期：2026-10-29。
