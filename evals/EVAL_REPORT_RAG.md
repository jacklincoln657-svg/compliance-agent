# 本地数据分析 Agent · RAG 硬路由 / ParentDocument 评测（E2）

> 脚本：`_eval_rag_suite.py`（不调用收费 LLM；需 chroma_db + 本地 embedding 缓存）  
> Query 数：**30** = 有 intended **24** + 无关/边界 **6**  
> Top-K = **6**

## 指标怎么读（先分清分母）

本评测是 **知识边界治理**，不是 RAG 召回质量 / 答对率。

| 指标 | 分母 | 含义 |
|------|------|------|
| 出现跨域 hit 的 query | 有 intended 的 24 | 至少 1 个 Top-K hit 的 category ≠ intended |
| 重度污染 query | 同上 24 | ≥半数 hit 跨域 |
| 跨域 hit 占比 | 有 intended 的全部 hit（144） | hit 级，不是 query 级 |
| 无关 6 条 | 不进跨域率分母 | 仅观察开放检索是否仍强行召回 |

## 对比定义

| 方案 | 含义 |
|------|------|
| **Before · 开放检索** | `similarity_search(q, k)` 无 category filter |
| **After · 硬路由** | `filter={"category": intended}` |

## 总指标

| 指标 | Before（开放） | After（硬路由） |
|------|----------------|-----------------|
| 出现跨域 hit 的 query | **16** / 24 | **0** / 24 |
| 重度污染 query（≥半数跨域） | **7** / 24 | **0** / 24 |
| 跨域 hit 占比（hit 级） | **29.2%**（42/144） | **0.0%**（0/144） |

换算备忘：全量 30 作分母时，出现跨域约为 16/30（53.3%）；无关条无 intended，简历优先写绝对次数或 intended 分母。

## 分桶摘要

| 桶 | 条数 | 开放跨域 hit | 锁定跨域 hit | 开放出现跨域的 query |
|----|------|--------------|--------------|----------------------|
| engineering | 8 | 15 | 0 | 5 |
| academic | 8 | 12 | 0 | 5 |
| cross | 8 | 15 | 0 | 6 |
| irrelevant | 6 | 0 | 0 | 0 |

## ParentDocument（机制行为验证 · 非效果 KPI）

在 6 条 probe 上：平均子块 hit **6.0** → unique parent **2.5**（去重成立=True）。

> 只证明子块召回后能聚到更少父块；不要写成比普通 chunk 更好，不要把均值写进简历。

| query | 域 | 子块数 | unique 父块 | 子块字符 | 父块字符 | 扩展比 |
|-------|----|--------|-------------|----------|----------|--------|
| 出仓脱敏 Phone Score 合规 | engineering | 6 | 2 | 923 | 1838 | 1.99× |
| 毕业设计 开题报告 研究内容 | academic | 6 | 2 | 439 | 1982 | 4.51× |
| 知识库硬路由 category | engineering | 6 | 3 | 368 | 2431 | 6.61× |
| 文献综述 国内外现状 | academic | 6 | 3 | 618 | 2982 | 4.83× |
| workspace 沙箱挂载 | engineering | 6 | 2 | 631 | 1478 | 2.34× |
| 论文进度安排 中期检查 | academic | 6 | 3 | 637 | 2982 | 4.68× |

## Query 明细

| 桶 | intended | 开放 cats | 锁定 cats | 出现跨域 | 重度污染 |
|----|----------|-----------|-----------|----------|----------|
| engineering | engineering | ['engineering', 'engineering', 'engineering', 'engineering', 'engineering', 'engineering'] | ['engineering', 'engineering', 'engineering', 'engineering', 'engineering', 'engineering'] |  |  |
| engineering | engineering | ['engineering', 'engineering', 'engineering', 'engineering', 'engineering', 'engineering'] | ['engineering', 'engineering', 'engineering', 'engineering', 'engineering', 'engineering'] |  |  |
| engineering | engineering | ['engineering', 'engineering', 'engineering', 'engineering', 'academic', 'engineering'] | ['engineering', 'engineering', 'engineering', 'engineering', 'engineering', 'engineering'] | Y |  |
| engineering | engineering | ['academic', 'engineering', 'academic', 'engineering', 'engineering', 'engineering'] | ['engineering', 'engineering', 'engineering', 'engineering', 'engineering', 'engineering'] | Y |  |
| engineering | engineering | ['engineering', 'engineering', 'engineering', 'engineering', 'engineering', 'engineering'] | ['engineering', 'engineering', 'engineering', 'engineering', 'engineering', 'engineering'] |  |  |
| engineering | engineering | ['academic', 'academic', 'engineering', 'academic', 'engineering', 'engineering'] | ['engineering', 'engineering', 'engineering', 'engineering', 'engineering', 'engineering'] | Y | Y |
| engineering | engineering | ['academic', 'engineering', 'academic', 'academic', 'engineering', 'engineering'] | ['engineering', 'engineering', 'engineering', 'engineering', 'engineering', 'engineering'] | Y | Y |
| engineering | engineering | ['academic', 'academic', 'academic', 'academic', 'academic', 'academic'] | ['engineering', 'engineering', 'engineering', 'engineering', 'engineering', 'engineering'] | Y | Y |
| academic | academic | ['academic', 'engineering', 'academic', 'engineering', 'academic', 'academic'] | ['academic', 'academic', 'academic', 'academic', 'academic', 'academic'] | Y |  |
| academic | academic | ['academic', 'academic', 'academic', 'academic', 'academic', 'academic'] | ['academic', 'academic', 'academic', 'academic', 'academic', 'academic'] |  |  |
| academic | academic | ['academic', 'engineering', 'academic', 'engineering', 'academic', 'academic'] | ['academic', 'academic', 'academic', 'academic', 'academic', 'academic'] | Y |  |
| academic | academic | ['academic', 'academic', 'academic', 'academic', 'academic', 'academic'] | ['academic', 'academic', 'academic', 'academic', 'academic', 'academic'] |  |  |
| academic | academic | ['engineering', 'engineering', 'engineering', 'academic', 'engineering', 'engineering'] | ['academic', 'academic', 'academic', 'academic', 'academic', 'academic'] | Y | Y |
| academic | academic | ['engineering', 'academic', 'academic', 'academic', 'academic', 'academic'] | ['academic', 'academic', 'academic', 'academic', 'academic', 'academic'] | Y |  |
| academic | academic | ['academic', 'academic', 'engineering', 'academic', 'engineering', 'academic'] | ['academic', 'academic', 'academic', 'academic', 'academic', 'academic'] | Y |  |
| academic | academic | ['academic', 'academic', 'academic', 'academic', 'academic', 'academic'] | ['academic', 'academic', 'academic', 'academic', 'academic', 'academic'] |  |  |
| cross | engineering | ['engineering', 'engineering', 'engineering', 'academic', 'engineering', 'engineering'] | ['engineering', 'engineering', 'engineering', 'engineering', 'engineering', 'engineering'] | Y |  |
| cross | academic | ['academic', 'engineering', 'academic', 'academic', 'academic', 'academic'] | ['academic', 'academic', 'academic', 'academic', 'academic', 'academic'] | Y |  |
| cross | engineering | ['engineering', 'engineering', 'engineering', 'engineering', 'engineering', 'engineering'] | ['engineering', 'engineering', 'engineering', 'engineering', 'engineering', 'engineering'] |  |  |
| cross | academic | ['academic', 'academic', 'academic', 'engineering', 'academic', 'academic'] | ['academic', 'academic', 'academic', 'academic', 'academic', 'academic'] | Y |  |
| cross | engineering | ['academic', 'academic', 'academic', 'engineering', 'academic', 'engineering'] | ['engineering', 'engineering', 'engineering', 'engineering', 'engineering', 'engineering'] | Y | Y |
| cross | academic | ['engineering', 'engineering', 'engineering', 'academic', 'engineering', 'engineering'] | ['academic', 'academic', 'academic', 'academic', 'academic', 'academic'] | Y | Y |
| cross | engineering | ['engineering', 'academic', 'engineering', 'engineering', 'academic', 'academic'] | ['engineering', 'engineering', 'engineering', 'engineering', 'engineering', 'engineering'] | Y | Y |
| cross | academic | ['academic', 'academic', 'academic', 'academic', 'academic', 'academic'] | ['academic', 'academic', 'academic', 'academic', 'academic', 'academic'] |  |  |
| irrelevant | — | ['engineering', 'engineering', 'academic', 'academic', 'engineering', 'engineering'] | — |  |  |
| irrelevant | — | ['engineering', 'academic', 'academic', 'engineering', 'academic', 'academic'] | — |  |  |
| irrelevant | — | ['engineering', 'engineering', 'academic', 'engineering', 'engineering', 'engineering'] | — |  |  |
| irrelevant | — | ['engineering', 'engineering', 'academic', 'engineering', 'academic', 'academic'] | — |  |  |
| irrelevant | — | ['academic', 'academic', 'academic', 'academic', 'engineering', 'academic'] | — |  |  |
| irrelevant | — | ['academic', 'academic', 'academic', 'academic', 'engineering', 'academic'] | — |  |  |

## 简历 / 口述表述（推荐 · 治理口径）

> 构造 **30** 条测试 query；在有目标域的 **24** 条上，开放检索有 **16** 条出现跨域命中，硬路由后为 **0**。（hit 级跨域占比 29.2%→0.0%，分母为有 intended 的 Top-K hit。）

**不要写**：RAG 准确率提升、ParentDocument 均值进简历、混淆 query 级与 hit 级分母。

## 局限

- intended 为构造标签，非大规模人工金标。
- 硬路由是确定性过滤器：锁库后跨域归零是治理目标，不是召回质量提升。
- ParentDocument 仅为机制验证。
