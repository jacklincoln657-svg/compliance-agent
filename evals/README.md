# evals/

本目录存放可复现的评测脚本与报告，用于验证合规硬闸、治理逻辑、RAG 硬路由与沙箱边界等能力。

请在**项目根目录**执行：

```bash
python evals/_eval_before_after.py      # → EVAL_REPORT.md（校验层 before/after）
python evals/_eval_governance.py        # → EVAL_REPORT_GOVERNANCE.md
python evals/_eval_rag_suite.py         # → EVAL_REPORT_RAG.md（需已建 chroma_db）
python evals/_eval_boundary.py          # → EVAL_REPORT_BOUNDARY.md
```

阅读报告时请注意：`EVAL_REPORT.md` 等对比的是**程序校验相对软检查的增益**，并不等价于大模型端到端任务准确率。
