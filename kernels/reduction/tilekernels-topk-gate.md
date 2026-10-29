# TileKernels MoE：小 k 的重复 argmax

每 token 一个 warp，重复选最大值并排除已选项，直接输出 expert 索引。

## 什么时候选择

- **适用**：有限 FP32 scores，小 num_topk，且 1≤k≤num_experts；并列分数按较小索引优先。
- **不适用**：k 接近专家数、大规模通用排序、需要概率和权重归一化的完整 router。
- **性能依据**：以下收益是源码机制分析，不代表已测得加速；没有本地 GPU 结果。

## 为什么这样设计

| 原来的问题 | 设计选择 | 收益与代价 |
| --- | --- | --- |
| 完整排序对小 k 浪费工作 | 重复 k 次 max+argmax | 避免全排序；成本随 k 线性增加 |
| 并列分数选择不稳定 | 等于最大值的索引取 min | 明确 tie-break；参考实现也要对齐 |
| 逐元素检查妨碍向量化 | 先载入，再统一归约 nonfinite 检查 | 保留向量载入机会；无效输入将触发断言 |

**下一步验证**：覆盖并列值、非整 warp 专家数、k 的边界；按专家数和 k 比较排序式 Top-k，明确交叉点。

## 核心完整代码

入口：`get_topk_gate_kernel_cuda`。完整工厂；输出只有索引，不包括 gate weight、softmax 或 token dispatch。

```python
import tilelang

from tilelang import language as T

from tile_kernels.utils import align

@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def get_topk_gate_kernel_cuda(num_experts: int, num_topk: int, use_pdl: bool = False):
    num_tokens = T.dynamic('num_tokens')
    num_threads = 32
    num_aligned_experts = align(num_experts, num_threads)

    @T.prim_func
    def topk_gate_kernel(
        scores: T.Tensor[(num_tokens, num_experts), T.float32],
        topk_idx: T.Tensor[(num_tokens, num_topk), T.int64],
    ):
        with T.Kernel(num_tokens, threads=num_threads) as pid:
            if use_pdl:
                T.pdl_sync()
            scores_fragment = T.alloc_fragment((num_aligned_experts,), T.float32)
            amax_fragment = T.alloc_fragment((1,), T.float32)
            idx_fragment = T.alloc_fragment((num_aligned_experts,), T.int32)
            idx_reducer = T.alloc_reducer((1,), T.int32, op='min')
            idx_result = T.alloc_fragment((1,), T.int32)
            nonfinite_reducer = T.alloc_reducer((1,), T.int32, op='sum')
            nonfinite_result = T.alloc_fragment((1,), T.int32)
            topk_idx_shared = T.alloc_shared((num_topk,), T.int32)

            for i in T.Parallel(num_aligned_experts):
                if i < num_experts:
                    scores_fragment[i] = scores[pid, i]
                else:
                    scores_fragment[i] = -T.infinity(T.float32)

            # Invalid input check, single deferred assert to keep the load loop vectorized
            T.reducer_init(nonfinite_reducer)
            for i in T.Parallel(num_aligned_experts):
                if i < num_experts and not T.isfinite(scores_fragment[i]):
                    T.reducer_update(nonfinite_reducer[0], 1)
            T.finalize_reducer(nonfinite_reducer, nonfinite_result)
            T.device_assert(nonfinite_result[0] == 0, msg='Score input for topk gate must be finite')

            for i in T.Parallel(num_aligned_experts):
                idx_fragment[i] = i

            # Get topk via repeatedly finding max
            for k in T.unroll(num_topk):
                T.reduce_max(scores_fragment, amax_fragment)
                T.reducer_init(idx_reducer)
                for i in T.Parallel(num_aligned_experts):
                    if scores_fragment[i] == amax_fragment[0]:
                        T.reducer_update(idx_reducer[0], idx_fragment[i])
                T.finalize_reducer(idx_reducer, idx_result)
                topk_idx_shared[k] = idx_result[0]
                for i in T.Parallel(num_aligned_experts):
                    if idx_fragment[i] == idx_result[0]:
                        scores_fragment[i] = -T.infinity(T.float32)

            T.copy(topk_idx_shared, topk_idx[pid, 0], disable_tma=True)
            if use_pdl:
                T.pdl_trigger()

    return topk_gate_kernel
```

## 来源与验证

- 来源：[deepseek-ai/TileKernels · topk_gate_cuda.py](https://github.com/deepseek-ai/TileKernels/blob/c1cf914a82e6901c679445f872051ce44ac2cc63/tile_kernels/moe/topk_gate_cuda.py)，commit `c1cf914a82e6901c679445f872051ce44ac2cc63`。
- 摘录范围：L1–L1, L2–L2, L4–L4, L7–L68。仅选取核心及其模块内依赖，保留函数体；去除与所选入口无关的外围代码。本文是摘录与分析，非上游原文件。
- 验证：[上游测试或参考入口](https://github.com/deepseek-ai/TileKernels/blob/c1cf914a82e6901c679445f872051ce44ac2cc63/tests/moe/test_topk_gate.py)；本轮只核查入口与源码，未执行 GPU 正确性、编译或性能测试。
- 代码版权与许可遵循[上游 LICENSE](../_licenses/tilekernels.txt)；代码中的原有版权声明保留。核查日期：2026-10-29。
