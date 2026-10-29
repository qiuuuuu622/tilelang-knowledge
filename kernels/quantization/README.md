# 量化与激活融合

先核对输出格式，而不是先比较微秒数：scale 的粒度、是否向上取 2 的幂、FP8/FP4 编码、舍入方式、物理 packing 与下游一致后，性能比较才有意义。

- 固定 BF16→FP8、每 128 元素一组：SGLang act quant。
- 多种 dtype/输入 scale/随机舍入：TileKernels per-token cast。
- scale 跨 token 共享：TileKernels per-block cast。
- 量化前还要 SwiGLU/路由缩放：优先检查融合案例的精确计算顺序。

## 案例索引

共 4 篇。代码行数用于估计阅读成本；先读“什么时候选择”和“为什么这样设计”，匹配后再读代码。

| 案例 | 值得借鉴的设计 | 核心代码行数 |
| --- | --- | ---: |
| [SGLang：128 元素分组的 FP8 激活量化](sglang-act-quant.md) | 把连续激活按 128 个元素求绝对值最大值、生成 scale 并写出 FP8；可选将 scale 向上取为 2 的幂。 | 119 |
| [TileKernels：可配置的 per-token 量化](tilekernels-per-token-cast.md) | 将分组 amax、scale 和类型转换组织为同一工厂，支持输入已有 scale 与可选随机舍入。 | 221 |
| [TileKernels：二维 block scale 量化](tilekernels-per-block-cast.md) | 在二维 token×channel 块内归约 amax 并量化，与 per-token 不同的是 scale 跨多个 token 共享。 | 142 |
| [TileKernels：SwiGLU 与量化融合](tilekernels-swiglu-quant.md) | 将 SwiGLU、可选路由权重/裁剪和分组量化串在一次数据处理里，直接产生低精度输出及 scale。 | 333 |

## 证据边界

全部条目已静态核查固定版本源码、保留完整选定函数及模块内依赖；未运行 GPU 编译、正确性或性能测试。上游测试入口不代表本仓库验证通过。复杂模型案例依赖对应上游工程，不能直接作为独立脚本运行。

[返回总索引](../README.md) · [来源、版本与收录边界](../SOURCES.md)
