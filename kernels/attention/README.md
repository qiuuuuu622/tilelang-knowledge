# Attention 与稀疏索引

先确定任务输出：最终 attention 向量、索引分数，还是池化后的 Top-k。三者不能互换。

- 学习完整 dense attention：先读官方前向，再读变长 GQA 与反向。
- MLA decode 并行度不足：对照官方 split-K 与 SGLang 稀疏 partial/combine，测包含合并的总时间。
- 已有稀疏索引：先用 DSA 基础版建立正确性，再考虑手动流水版。
- 要生成稀疏索引：MiniCPM pooling、QSA 打分、FP8 paged MQA 各自有模型语义，不能按形状互相替换。
- 已有 DSV4 打包 cache：读取专用案例；相同 dtype/shape 不代表相同字节布局。

## 案例索引

共 11 篇。代码行数用于估计阅读成本；先读“什么时候选择”和“为什么这样设计”，匹配后再读代码。

| 案例 | 值得借鉴的设计 | 核心代码行数 |
| --- | --- | ---: |
| [SGLang MiniCPM：attention pooling + Top-k](sglang-minicpm-pooling-topk.md) | 计算压缩 attention 的块选择分数与索引，将归一化、pooling 和在线 Top-k 连在一起；输出并非最终 attention 向量。 | 415 |
| [SGLang DSA：稀疏 gather + online softmax](sglang-dsa-sparse-baseline.md) | 只遍历索引指定的 KV，在线维护 softmax 与输出，是理解 DSA 稀疏 attention 的较短实现。 | 204 |
| [SGLang DSA：手动流水与线程分工](sglang-dsa-warp-specialized.md) | 以 384 线程分担 KV 装载、分数计算和左右输出累积，重叠稀疏数据准备与计算。 | 409 |
| [SGLang DSA：稀疏 decode 的分段与合并](sglang-dsa-split-decode.md) | 把选中 KV 分成多个 CTA 计算局部输出和 LSE，再按 softmax 权重合并，提高小 query 场景的并行度。 | 297 |
| [SGLang：FP8 paged MQA 索引打分](sglang-fp8-paged-mqa-logits.md) | 从分页 KV cache 生成 MQA 索引分数，供后续稀疏选择使用。 | 175 |
| [SGLang QSA：prefill/decode 索引打分分工](sglang-qsa-mqa.md) | 按 query 数量区分 prefill 与 decode 的 MQA 打分组织方式，服务 QSA 索引器。 | 457 |
| [SGLang DSV4：打包 FP8 cache 的稀疏 attention](sglang-dsv4-fp8-cache-attention.md) | 在稀疏 KV 读取时解读打包的 FP8 数据、scale 与尾部，并通过 partial/combine 完成 attention。 | 1050 |
| [TileLang 官方：FlashAttention 前向](tilelang-flashattention-forward.md) | 对 BSHD 布局的 Q/K/V 分块，在线归一化并累积输出，避免生成完整 S×S 概率矩阵。 | 91 |
| [TileLang 官方：变长 GQA 与右对齐 causal](tilelang-gqa-varlen.md) | 用累计长度索引 packed 序列，复用较少的 KV heads，并处理 Q/K 长度不等的右对齐 causal mask。 | 124 |
| [TileLang 官方：MLA decode 的 split-K](tilelang-mla-decode.md) | 把 MLA decode 的长 KV 序列切成多份，用局部输出与 LSE 合并扩大并行度。 | 178 |
| [TileLang 官方：FlashAttention 前后向闭环](tilelang-flashattention-backward.md) | 保存前向 LSE，在反向重算分块概率，计算 dQ/dK/dV，避免保存完整注意力矩阵。 | 275 |

## 证据边界

全部条目已静态核查固定版本源码、保留完整选定函数及模块内依赖；未运行 GPU 编译、正确性或性能测试。上游测试入口不代表本仓库验证通过。复杂模型案例依赖对应上游工程，不能直接作为独立脚本运行。

[返回总索引](../README.md) · [来源、版本与收录边界](../SOURCES.md)
