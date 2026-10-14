# 本地数据分析 Agent · 治理/RAG/沙箱评测报告

> 脚本：`_eval_governance.py`（不调用收费 LLM API）
> 硬校验 before/after 见 `EVAL_REPORT.md` / `_eval_before_after.py`

## 总览

| 能力点 | 结果 | 说明 |
|--------|------|------|
| Tool轨迹清洗 | PASS | 字符量 12330→609，降幅 **95%** |
| 弹性摘要窗口 | PASS | Max=8/Keep=4；至约14轮摘要调用 **2** 次（若过Keep每轮摘要约 **10** 次） |
| COW回滚 | PASS | 恢复旧文件=True，删除新建=True (restored=1, deleted=1) |
| RAG硬路由 | PASS | 无filter串到非engineering **5/6**；锁engineering后串库 **0**；unique parent_id=3 |
| 沙箱后端 | PASS | 当前 **local**（docker可用=False）—此项为能力探测，不是 before/after 准确率；Docker 可用时简历可写「实测优先容器」。 |

## 与简历三条 bullet 的对应

| 简历表述 | 本评测能量化的部分 | 暂不能装成「准确率%」的部分 |
|----------|-------------------|------------------------------|
| 可验证交付与合规硬闸 | → 见 `EVAL_REPORT.md`（软检查 vs 硬闸） | Checklist/脚本自检仍是 Prompt 软约束 |
| Policy RAG + 硬路由 | 有/无 category filter 串库次数；parent_id 去重规模 | 「规范约束了 Agent 行为」需端到端 LLM 评测 |
| 受控执行与状态治理 | COW 回滚成败；tool清洗降幅；弹性摘要调用次数；Docker是否可用 | Docker vs local 的逃逸对比需额外攻击用例 |

## 简历可引用的诚实句子（示例）

1. Tool轨迹清洗：合成多轮历史字符量下降约 **95%**。
2. 弹性窗口：模拟至约14用户轮，摘要 LLM 调用 **2** 次（对照「过Keep每轮摘要」估约 **10** 次）。
3. COW：注入改旧+新建后 rollback，旧内容恢复且新建删除（本机合成用例 **通过**）。
4. 知识硬路由：查询偏任务书语义时，无过滤串出 engineering 外类别 **5** 次，锁定 engineering 后串库 **0** 次。
5. 沙箱：本机探测后端为 **local**（当前回退 local，勿写已强制容器隔离）。

## 局限

- 未跑真实 DeepSeek 多轮 Agent，故无「端到端任务成功率」。
- RAG 数字依赖当前 `chroma_db` 语料；换库需重跑。
- Checklist 层无法用本脚本单独量化。
