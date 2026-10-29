# P03 选择工作粒度

**区分“可运行 CTA 太少”和“CTA 内可运行 warp 不足”，再选择拆分或合并工作。**

## 触发证据

- 小 batch/decode 只产生少量 CTA，许多 SM/CU 无任务。
- CTA 很小，驻留 block 数先达到限制，仍缺少用于隐藏延迟的活跃 warp。
- 相邻 shape 突然变慢，启动几何或整除分支发生跳变。

低 occupancy 本身不证明瓶颈。结合网格、寄存器/shared 限制、等待原因与完整延迟判断。

## 决策与 TileLang 落点

| 原因 | 最小候选 | 代价 |
| --- | --- | --- |
| 独立输出维度尚未展开 | 沿 hidden/head/output tile 划分 CTA | 重复加载、尾部处理 |
| 独立输出不足 | 沿归约维度 split，输出 partial 后 combine | 临时缓冲、额外 launch、合并与数值次序变化 |
| block 驻留上限先受限 | 扫描每 CTA 的 warp/线程数 | 更大 block 可能降低调度灵活性或增加资源 |
| 调度整除导致极小 block | 稳定启动几何并 mask 尾部 | 所有参与 barrier 的线程必须继续参与同步 |

TileLang：联合检查 `T.Kernel(..., threads=...)`、tile 和 fragment 布局。split attention 的 partial 必须携带正确归一化信息，不能直接平均各段输出。

完整代码：[DSA partial/combine](../../kernels/attention/sglang-dsa-split-decode.md)、[MLA decode](../../kernels/attention/tilelang-mla-decode.md)。

## 证据与反例

- **上游测量 / TileLang**：[S07](../SOURCES.md#s07) MI355、64 heads，query 长度 1 时 sparse decode 由 0.339→0.029 ms；query 长度 2048 时则由 3.135→6.600 ms。最终仅在小 query 分支启用。kernel 加速不能等同模型吞吐提升。
- **消融与最终 diff / CUDA**：[S08](../SOURCES.md#s08) B200 indexer 量化最终只保留 block size 128→256。persistent grid-stride 在部分大 batch 上比仅放大 CTA 慢约 3%，随后移除；PR 首段仍描述过时的改动，应以最终 diff 与审查收尾为准。
- **源码事实与上游测量 / CUDA**：[S09](../SOURCES.md#s09) 量化奇数行继承接近偶数行的启动策略，再做 bounds check；没有实际补齐/拷贝输入。H100 指定 M=13359、K=2688、group=128 配置，报告热缓存 339.63→81.46 μs；不是端到端模型测量。

## 最小验证与退出

1. 分开扫描 CTA 大小与 split 数，不同时修改算法、精度和缓存策略。
2. 计入 partial 分配、combine 和 launch；测试小/大 batch、非整形状、全 mask 段。
3. 检查数值容差与稳定性；对 persistent 额外做“同 CTA、无 persistent”的消融。

**退出**：combine/重复加载成本超过并行收益；资源占用使可并行工作减少；新配置只改善某个计数器。保留小负载专用分派，不把胜出配置设成跨 GPU 常量。

[模式索引](../README.md)
