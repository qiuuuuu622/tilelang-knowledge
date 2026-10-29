# 复制完成、线程同步与流水依赖

> 关键词：T.copy / T.async_copy / T.tma_copy / ptx_wait_group / T.Pipelined。
> 适用：普通复制通用概念；cp.async、TMA 等显式协议为 CUDA 专属且要求相应架构；基线 `0a8c6716513b`，见[版本与来源](../SOURCES.md)。
> 核验：已核查所列上游资料；本文代码仅做 Python 语法检查，未做 TileLang 编译或 GPU 实测。

**“已发出搬运”“搬运已完成”“消费者可以安全读取”“缓冲可以重用”必须分别判断。**

## 语义与约束

| 操作 | 本快照的高层契约 | 用户应检查什么 |
| --- | --- | --- |
| `T.copy` tile 搬运 | 普通复制表达依赖；具体指令与必要同步由 lowering 处理 | 不从 API 名字推断必定使用 SIMT、cp.async 或 TMA |
| `T.async_copy` 的 global→shared tile 路径 | 发出 cp.async 与 commit；不自动插入 async wait | 消费前显式安排 `T.ptx_wait_group`；另检查跨线程可见性 |
| `T.tma_copy` global→shared | 显式 TMA 生产侧操作，不包含完整等待 | barrier、事务字节、arrive/wait、phase 与参与线程 |
| `T.tma_copy` shared→global | 发出 store 与 arrive，不包含 store wait | 正确安排 `T.tma_store_wait` 与源缓冲复用 |

异步完成等待不能无条件替代线程间 barrier。标准 ThreadSync 可插入 shared 同步，但手工线程角色分工、分支或关闭 pass 后必须重新验证；不要随手关闭检查以换取“编译成功”。

## 最小检查

先运行 [普通复制示例](../api/copy-and-views.md)。显式异步路径参考下方固定版本的 SM80 测试：它在两个 `T.async_copy` 后分别插入 `T.ptx_wait_group(0)`，比较与普通复制的生成代码。**该测试是特定配置的 codegen 对照，不是性能提升证明，也不是所有 GPU 上的等价保证。**

流水化优先使用 `T.Pipelined(..., num_stages=s)`；需要手工 `stage/order` 时，两组数组对应被调度的可执行语句，不为可重放的标量 Bind 别名单独计项。通常不再同时传 `num_stages`。

## 为什么这样设计

显式异步给出更多调度空间，也把完成等待与缓冲生命周期责任交给实现者。立刻等待往往便于建立正确基线；是否把等待后移，需要由依赖与重叠实验决定。

## 常见误用与检查

| 症状 | 检查动作 |
| --- | --- |
| 偶发错误或挂起 | 逐个列出生产、消费、释放；审查所有线程分支及 barrier 参与者 |
| 第一次正确、多轮错误 | 检查 ping-pong 索引、phase 和跨迭代覆写 |
| 增 stage 后错误 | 退回 serial，覆盖 1/2/3 个 tile、多轮与尾部；保留失败配置 |
| 只加 block barrier 仍错误 | 检查异步操作自身是否已完成，不能只看线程是否集合 |

## 来源与关联

- [tilelang/language/copy_op.py](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/tilelang/language/copy_op.py)
- [testing/python/language/test_tilelang_language_async_copy_gemm_sm80.py](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/testing/python/language/test_tilelang_language_async_copy_gemm_sm80.py)
- [docs/programming_guides/software_pipeline.md](https://github.com/tile-ai/tilelang/blob/0a8c6716513bcdf8a728735c531c210c33077784/docs/programming_guides/software_pipeline.md)
- [流水资源模式](../../patterns/pipelining/budget-pipeline-resources.md) · [竞争与复现](../debugging/races-and-reproducers.md)
