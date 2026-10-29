# P05 在明确边界内融合

**只有中间结果的所有用途、舍入与同步要求都被保留，融合减少的读写和启动才有意义。**

## 触发证据

时间线显示相邻小 kernel 反复读写相同中间张量；或者 producer 已持有 consumer 所需的数据，却先写回再读取。先列出所有消费者，包括 residual、辅助输出、量化 scale 和模型旁路。

## 决策与 TileLang 落点

1. 标出可以直接消费的中间值，以及必须落地供其他路径读取的值。
2. 保持完整数学语义：clamp、mask、scale、归约、epsilon、舍入点和输出格式。
3. 只融合一条明确的计算链，控制活跃 fragment/shared 大小；保留不适用 shape 的分派回退。
4. 先调优未融合基线，避免把基线参数不佳误当融合收益。

TileLang：在 fragment 中复用已产生的结果，必要时落地 shared；不要默认“保留 FP32”与“先写 BF16 再读回”逐位等价。

完整代码：[vLLM post/pre](../../kernels/normalization/vllm-mhc-fused-post-pre.md)、[SGLang post/pre](../../kernels/normalization/sglang-mhc-fused-post-pre.md)、[SwiGLU+quant](../../kernels/quantization/tilekernels-swiglu-quant.md)。

## 证据与反例

- **上游测量 / TileLang**：[S10](../SOURCES.md#s10) GB300、hc=4、hidden=5120，以调优后的未融合链为基线：1 token 为 11.09→8.44 μs；32 为 11.62→11.29 μs；48 为 11.72→13.55 μs。最终限制融合范围。多个 kernel 的整条 seam 与单 kernel 的时间不能混用。
- **消费者边界**：同一 PR 的 Engram 分支要读取 post 与 pre 之间的 residual，因此保留独立执行。最后一层和辅助状态采集也要单独核对。
- **数值边界**：同一实现使用寄存器中的 FP32 post 值投影，而未融合路径使用 BF16 舍入结果；必须按明确参考与容差验证，不能因为 stored residual 相同就假设全部输出相同。
- **反例 / Triton 快路径**：[S11](../SOURCES.md#s11) 为 SwiGLU clamp 非空时禁用 fused SiLU+block-quant。省略 clamp 的实现不是等价融合；该修复只提供迁移约束，不是 TileLang 加速证据。

## 最小验证与退出

1. 对照输入到所有输出，覆盖 clamp、极值、旁路、辅助输出和 shape 回退。
2. 验证数值差异来自何处；不能为了通过测试盲目放宽容差。
3. 测调优后的完整未融合链与融合链，记录资源、临时缓冲和模型层面的影响。

**退出**：中间值必须被外部消费且无法保持时序；数值约定不满足；寄存器/shared 开销导致完整链变慢。优先缩小融合范围或限制适用负载。

[模式索引](../README.md)
