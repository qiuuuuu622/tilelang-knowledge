# TileKernels mHC：小矩阵 Sinkhorn 前后向

把小 mixing 矩阵的多轮行列归一化放在片上，反向重放中间归一化状态。

## 什么时候选择

- **适用**：FP32 小方阵、repeat≥1，与 mHC 的 epsilon 添加位置一致；训练需要匹配前后向定义。
- **不适用**：把有限轮 Sinkhorn 当成严格双随机矩阵；将 epsilon 随意移到不同步骤。
- **性能依据**：以下收益是源码机制分析，不代表已测得加速；没有本地 GPU 结果。

## 为什么这样设计

| 原来的问题 | 设计选择 | 收益与代价 |
| --- | --- | --- |
| 每轮行列归一化都 launch 成本高 | fragment 内循环完成 repeat | 减少 launch/全局中间写回；迭代串行 |
| 指数可能溢出 | 先减行最大值再 exp | 改善数值范围；极值仍需测试 |
| 反向需要各轮归一化状态 | 重放并保存到 shared 的 xs/sums | 避免上层长期保存；共享内存随 repeat 和块大小增长 |

**下一步验证**：检验行列和及梯度误差随 repeat/eps 的变化，再比较小 token 与批量 token 的 token_block_size。

## 核心完整代码

入口：`mhc_sinkhorn_fwd_cuda`, `mhc_sinkhorn_bwd_cuda`。完整前后向。反向默认 token_block_size=32，前向默认 1；两个默认值不能当成普遍最优配置。

```python
import tilelang

from tilelang import language as T

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}

@tilelang.jit(pass_configs=_PASS_CONFIGS)
def mhc_sinkhorn_fwd_cuda(
    comb_res_mix,
    comb_res_mix_out,
    repeat: int,
    eps: float,
    token_block_size: int = 1,
    use_pdl: bool = False,
):
    num_tokens = T.dynamic('num_tokens')
    hidden_size = T.const('hidden_size')

    comb_res_mix: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]
    comb_res_mix_out: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]

    with T.Kernel(T.ceildiv(num_tokens, token_block_size)) as pid_x:
        if use_pdl:
            T.pdl_sync()
        comb_frag = T.alloc_fragment((token_block_size, hidden_size, hidden_size), T.float32)
        row_sum = T.alloc_fragment((token_block_size, hidden_size), T.float32)
        col_sum = T.alloc_fragment((token_block_size, hidden_size), T.float32)

        T.copy(comb_res_mix[pid_x * token_block_size, 0, 0], comb_frag)

        row_max = T.alloc_fragment((token_block_size, hidden_size), T.float32)
        T.reduce_max(comb_frag, row_max, dim=2)
        for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
            comb_frag[i, j, k] = T.exp(comb_frag[i, j, k] - row_max[i, j])
        T.reduce_sum(comb_frag, row_sum, dim=2)
        for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
            comb_frag[i, j, k] = comb_frag[i, j, k] / row_sum[i, j] + eps

        T.reduce_sum(comb_frag, col_sum, dim=1)
        for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
            comb_frag[i, j, k] = comb_frag[i, j, k] / (col_sum[i, k] + eps)

        for _ in T.serial(repeat - 1):
            T.reduce_sum(comb_frag, row_sum, dim=2)
            for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
                comb_frag[i, j, k] = comb_frag[i, j, k] / (row_sum[i, j] + eps)

            T.reduce_sum(comb_frag, col_sum, dim=1)
            for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
                comb_frag[i, j, k] = comb_frag[i, j, k] / (col_sum[i, k] + eps)

        T.copy(comb_frag, comb_res_mix_out[pid_x * token_block_size, 0, 0])
        if use_pdl:
            T.pdl_trigger()

@tilelang.jit(pass_configs=_PASS_CONFIGS)
def mhc_sinkhorn_bwd_cuda(
    grad_output,
    x,
    grad_input,
    repeat: int,
    eps: float,
    token_block_size: int = 32,
    use_pdl: bool = False,
):
    num_tokens = T.dynamic('num_tokens')
    hidden_size = T.const('hidden_size')

    grad_output: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]
    x: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]
    grad_input: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32]

    with T.Kernel(T.ceildiv(num_tokens, token_block_size)) as pid_x:
        if use_pdl:
            T.pdl_sync()
        grad_frag = T.alloc_fragment((token_block_size, hidden_size, hidden_size), T.float32)
        x_frag = T.alloc_fragment((token_block_size, hidden_size, hidden_size), T.float32)

        T.copy(grad_output[pid_x * token_block_size, 0, 0], grad_frag)
        T.copy(x[pid_x * token_block_size, 0, 0], x_frag)

        row_sum = T.alloc_fragment((token_block_size, hidden_size), T.float32)
        row_sum2 = T.alloc_fragment((token_block_size, hidden_size), T.float32)
        col_sum = T.alloc_fragment((token_block_size, hidden_size), T.float32)
        col_sum2 = T.alloc_fragment((token_block_size, hidden_size), T.float32)
        temp = T.alloc_fragment((token_block_size, hidden_size, hidden_size), T.float32)

        xs = T.alloc_shared((repeat * 2, token_block_size, hidden_size, hidden_size), T.float32)
        sums = T.alloc_shared((repeat * 2, token_block_size, hidden_size), T.float32)

        row_max = T.alloc_fragment((token_block_size, hidden_size), T.float32)
        T.reduce_max(x_frag, row_max, dim=2)
        for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
            x_frag[i, j, k] = T.exp(x_frag[i, j, k] - row_max[i, j])
        T.reduce_sum(x_frag, row_sum, dim=2)
        for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
            x_frag[i, j, k] = x_frag[i, j, k] / row_sum[i, j]
        T.copy(x_frag, xs[0, 0, 0, 0])
        for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
            x_frag[i, j, k] = x_frag[i, j, k] + eps
        T.copy(x_frag, xs[1, 0, 0, 0])

        T.reduce_sum(x_frag, col_sum, dim=1)
        T.copy(col_sum, sums[1, 0, 0])
        for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
            x_frag[i, j, k] = x_frag[i, j, k] / (col_sum[i, k] + eps)

        for step in T.unroll(repeat - 1):
            T.reduce_sum(x_frag, row_sum, dim=2)
            T.copy(row_sum, sums[step * 2 + 2, 0, 0])
            T.copy(x_frag, xs[step * 2 + 2, 0, 0, 0])
            for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
                x_frag[i, j, k] = x_frag[i, j, k] / (row_sum[i, j] + eps)

            T.reduce_sum(x_frag, col_sum, dim=1)
            T.copy(col_sum, sums[step * 2 + 3, 0, 0])
            T.copy(x_frag, xs[step * 2 + 3, 0, 0, 0])
            for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
                x_frag[i, j, k] = x_frag[i, j, k] / (col_sum[i, k] + eps)

        x_inter = T.alloc_fragment((token_block_size, hidden_size, hidden_size), T.float32)
        for inv_step in T.unroll(2 * repeat - 1):
            T.copy(xs[2 * repeat - 1 - inv_step, 0, 0, 0], x_inter)
            if inv_step % 2 == 0:
                T.copy(sums[2 * repeat - 1 - inv_step, 0, 0], col_sum)
                for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
                    temp[i, j, k] = grad_frag[i, j, k] * x_inter[i, j, k]
                T.reduce_sum(temp, col_sum2, dim=1)
                for i, k in T.Parallel(token_block_size, hidden_size):
                    col_sum2[i, k] /= col_sum[i, k] + eps
                for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
                    grad_frag[i, j, k] = (grad_frag[i, j, k] - col_sum2[i, k]) / (col_sum[i, k] + eps)
            else:
                T.copy(sums[2 * repeat - 1 - inv_step, 0, 0], row_sum)
                for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
                    temp[i, j, k] = grad_frag[i, j, k] * x_inter[i, j, k]
                T.reduce_sum(temp, row_sum2, dim=2)
                for i, j in T.Parallel(token_block_size, hidden_size):
                    row_sum2[i, j] /= row_sum[i, j] + eps
                for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
                    grad_frag[i, j, k] = (grad_frag[i, j, k] - row_sum2[i, j]) / (row_sum[i, j] + eps)

        T.copy(xs[0, 0, 0, 0], x_inter)
        for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
            temp[i, j, k] = grad_frag[i, j, k] * x_inter[i, j, k]
        T.reduce_sum(temp, row_sum, dim=2)
        for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
            grad_frag[i, j, k] = (grad_frag[i, j, k] - row_sum[i, j]) * x_inter[i, j, k]

        T.copy(grad_frag, grad_input[pid_x * token_block_size, 0, 0])
        if use_pdl:
            T.pdl_trigger()
```

## 来源与验证

- 来源：[deepseek-ai/TileKernels · sinkhorn_cuda.py](https://github.com/deepseek-ai/TileKernels/blob/c1cf914a82e6901c679445f872051ce44ac2cc63/tile_kernels/mhc/sinkhorn_cuda.py)，commit `c1cf914a82e6901c679445f872051ce44ac2cc63`。
- 摘录范围：L1–L1, L2–L2, L4–L6, L9–L56, L59–L155。仅选取核心及其模块内依赖，保留函数体；去除与所选入口无关的外围代码。本文是摘录与分析，非上游原文件。
- 验证：[上游测试或参考入口](https://github.com/deepseek-ai/TileKernels/blob/c1cf914a82e6901c679445f872051ce44ac2cc63/tests/mhc/test_sinkhorn.py)；本轮只核查入口与源码，未执行 GPU 正确性、编译或性能测试。
- 代码版权与许可遵循[上游 LICENSE](../_licenses/tilekernels.txt)；代码中的原有版权声明保留。核查日期：2026-10-29。
