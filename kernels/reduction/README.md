# Softmax 与 Top-k

Top-k 与 softmax 的核心取舍不同：小 k 可比较重复 argmax 与排序；softmax 则先看行能否常驻片上，以及行数是否足以隐藏重读延迟。不要把当前行最大值、全部行最大值和跨分段 LSE 的归并混为一谈。

## 案例索引

共 3 篇。代码行数用于估计阅读成本；先读“什么时候选择”和“为什么这样设计”，匹配后再读代码。

| 案例 | 值得借鉴的设计 | 核心代码行数 |
| --- | --- | ---: |
| [TileLang 官方：长行 online softmax](tilelang-online-softmax.md) | 第一遍按块更新 log-sum-exp，第二遍重读输入并输出归一化结果，限制片上状态大小。 | 47 |
| [TileKernels MoE：小 k 的重复 argmax](tilekernels-topk-gate.md) | 每 token 一个 warp，重复选最大值并排除已选项，直接输出 expert 索引。 | 68 |
| [TileOPs：softmax 常驻与流式读取](tileops-softmax-resident-streaming.md) | 对可常驻的行直接归一化，对超长行以在线 max/sum 两遍扫描，并保留部分头部数据减少重读。 | 339 |

## 证据边界

全部条目已静态核查固定版本源码、保留完整选定函数及模块内依赖；未运行 GPU 编译、正确性或性能测试。上游测试入口不代表本仓库验证通过。复杂模型案例依赖对应上游工程，不能直接作为独立脚本运行。

[返回总索引](../README.md) · [来源、版本与收录边界](../SOURCES.md)
