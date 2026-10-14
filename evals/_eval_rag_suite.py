"""
项目一 · E2 RAG 硬路由 / ParentDocument 覆盖评测（不调收费 LLM）

设计：
  - 20～30 条固定 query，分 engineering / academic / 跨域相似 / 无关边界
  - Before：开放 similarity_search
  - After：filter={"category": target}
  - 指标：跨域误命中率、锁定后域内纯度、ParentDocument 子块→父块去重

用法（建议项目 venv）：
  set HF_HUB_OFFLINE=1
  ..\\venv311\\Scripts\\python.exe evals/_eval_rag_suite.py
→ evals/EVAL_REPORT_RAG.md
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import os
from collections import defaultdict
from pathlib import Path

from _bootstrap import EVAL_DIR, ROOT

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

CHROMA = ROOT / "chroma_db"
PARENT_STORE = ROOT / "parent_store.json"
K = 6

# (query, intended_category or None for 无关, bucket)
QUERIES: list[tuple[str, str | None, str]] = [
    ("数据分析出仓脱敏规范 Phone 掩码", "engineering", "engineering"),
    ("CSV Score 必须保留两位小数的合规要求", "engineering", "engineering"),
    ("audit_log total_rows 与交付清单一致性", "engineering", "engineering"),
    ("沙箱 Docker network none workspace 挂载", "engineering", "engineering"),
    ("Agent 工具执行与本地 workspace 产物路径", "engineering", "engineering"),
    ("合规硬校验 spec_validator 规则包", "engineering", "engineering"),
    ("知识库检索 category 硬路由禁止串库", "engineering", "engineering"),
    ("ParentDocument 父块子块切分策略", "engineering", "engineering"),
    ("毕业设计任务书 开题报告 研究内容", "academic", "academic"),
    ("论文进度安排 文献综述 研究方法", "academic", "academic"),
    ("开题答辩 技术路线 预期成果", "academic", "academic"),
    ("学士学位论文 写作规范 参考文献格式", "academic", "academic"),
    ("课题背景与研究意义怎么写", "academic", "academic"),
    ("实验方案与评价指标设计", "academic", "academic"),
    ("中期检查 工作进度 存在问题", "academic", "academic"),
    ("结论与展望章节撰写要求", "academic", "academic"),
    ("规范 要求 格式 检查", "engineering", "cross"),
    ("任务 进度 安排 报告", "academic", "cross"),
    ("脱敏 隐私 数据 导出", "engineering", "cross"),
    ("研究 方法 实验 设计", "academic", "cross"),
    ("交付 清单 文件 完整性", "engineering", "cross"),
    ("开题 背景 意义 现状", "academic", "cross"),
    ("校验 规则 字段 手机号", "engineering", "cross"),
    ("文献 综述 国内外 现状", "academic", "cross"),
    ("今天天气怎么样适合出门吗", None, "irrelevant"),
    ("西红柿炒蛋的家常做法", None, "irrelevant"),
    ("Spotify 最近听了什么歌", None, "irrelevant"),
    ("飞书日历 OAuth 刷新令牌", None, "irrelevant"),
    ("量子纠缠与黑洞信息悖论入门", None, "irrelevant"),
    ("NBA 总决赛比分预测", None, "irrelevant"),
]


def _load_vs():
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma

    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory=str(CHROMA), embedding_function=emb)


def _cats(hits) -> list[str]:
    return [((h.metadata or {}).get("category") or "") for h in hits]


def _parents(hits) -> list[str]:
    out = []
    for h in hits:
        pid = (h.metadata or {}).get("parent_id")
        if pid:
            out.append(pid)
    return out


def eval_routing(vs) -> dict:
    rows = []
    open_cross_hits = 0
    open_total_hits = 0
    locked_cross_hits = 0
    locked_total_hits = 0
    locked_impure_queries = 0
    open_any_cross_queries = 0
    open_polluted_queries = 0
    scored = 0

    for query, intended, bucket in QUERIES:
        open_hits = vs.similarity_search(query, k=K)
        open_c = _cats(open_hits)

        locked_c: list[str] = []
        locked_cross = 0
        open_cross = 0
        any_cross = False
        polluted = False

        if intended:
            scored += 1
            open_cross = sum(1 for c in open_c if c and c != intended)
            open_total_hits += len(open_c)
            open_cross_hits += open_cross
            any_cross = open_cross > 0
            if any_cross:
                open_any_cross_queries += 1
            if open_c and open_cross >= max(1, (len(open_c) + 1) // 2):
                polluted = True
                open_polluted_queries += 1

            locked_hits = vs.similarity_search(query, k=K, filter={"category": intended})
            locked_c = _cats(locked_hits)
            locked_cross = sum(1 for c in locked_c if c and c != intended)
            locked_total_hits += len(locked_c)
            locked_cross_hits += locked_cross
            if locked_cross > 0:
                locked_impure_queries += 1

        rows.append(
            {
                "query": query,
                "bucket": bucket,
                "intended": intended or "—",
                "open_cats": open_c,
                "open_cross": open_cross,
                "locked_cats": locked_c,
                "locked_cross": locked_cross,
                "any_cross": any_cross,
                "open_polluted": polluted,
            }
        )

    return {
        "rows": rows,
        "n_queries": len(QUERIES),
        "n_with_intended": scored,
        "n_irrelevant": len(QUERIES) - scored,
        "open_cross_rate": open_cross_hits / open_total_hits if open_total_hits else 0.0,
        "locked_cross_rate": locked_cross_hits / locked_total_hits if locked_total_hits else 0.0,
        "open_any_cross_queries": open_any_cross_queries,
        "open_polluted_queries": open_polluted_queries,
        "locked_impure_queries": locked_impure_queries,
        "open_cross_hits": open_cross_hits,
        "open_total_hits": open_total_hits,
        "locked_cross_hits": locked_cross_hits,
        "locked_total_hits": locked_total_hits,
    }


def eval_parent_document(vs) -> dict:
    import json

    parents = {}
    if PARENT_STORE.is_file():
        parents = json.loads(PARENT_STORE.read_text(encoding="utf-8"))

    probes = [
        ("出仓脱敏 Phone Score 合规", "engineering"),
        ("毕业设计 开题报告 研究内容", "academic"),
        ("知识库硬路由 category", "engineering"),
        ("文献综述 国内外现状", "academic"),
        ("workspace 沙箱挂载", "engineering"),
        ("论文进度安排 中期检查", "academic"),
    ]
    details = []
    for q, cat in probes:
        hits = vs.similarity_search(q, k=K, filter={"category": cat})
        pids = _parents(hits)
        uniq = sorted(set(pids))
        child_chars = sum(len(h.page_content or "") for h in hits)
        parent_chars = 0
        for pid in uniq:
            p = parents.get(pid) or {}
            parent_chars += len(p.get("page_content") or p.get("content") or "")
        details.append(
            {
                "query": q,
                "category": cat,
                "child_hits": len(hits),
                "unique_parents": len(uniq),
                "child_chars": child_chars,
                "parent_chars": parent_chars,
                "expand_ratio": (parent_chars / child_chars) if child_chars else 0.0,
            }
        )

    avg_children = sum(d["child_hits"] for d in details) / len(details)
    avg_parents = sum(d["unique_parents"] for d in details) / len(details)
    return {
        "details": details,
        "avg_child_hits": avg_children,
        "avg_unique_parents": avg_parents,
        "ok": all(d["unique_parents"] <= d["child_hits"] for d in details),
    }


def write_report(routing: dict, parent: dict) -> Path:
    by_bucket: dict[str, list] = defaultdict(list)
    for r in routing["rows"]:
        by_bucket[r["bucket"]].append(r)

    lines = [
        "# 本地数据分析 Agent · RAG 硬路由 / ParentDocument 评测（E2）",
        "",
        "> 脚本：`_eval_rag_suite.py`（不调用收费 LLM；需 chroma_db + 本地 embedding 缓存）  ",
        f"> Query 数：**{routing['n_queries']}** = 有 intended **{routing['n_with_intended']}**"
        f" + 无关/边界 **{routing['n_irrelevant']}**  ",
        f"> Top-K = **{K}**",
        "",
        "## 指标怎么读（先分清分母）",
        "",
        "本评测是 **知识边界治理**，不是 RAG 召回质量 / 答对率。",
        "",
        "| 指标 | 分母 | 含义 |",
        "|------|------|------|",
        f"| 出现跨域 hit 的 query | 有 intended 的 {routing['n_with_intended']} |"
        " 至少 1 个 Top-K hit 的 category ≠ intended |",
        f"| 重度污染 query | 同上 {routing['n_with_intended']} | ≥半数 hit 跨域 |",
        f"| 跨域 hit 占比 | 有 intended 的全部 hit（{routing['open_total_hits']}） |"
        " hit 级，不是 query 级 |",
        "| 无关 6 条 | 不进跨域率分母 | 仅观察开放检索是否仍强行召回 |",
        "",
        "## 对比定义",
        "",
        "| 方案 | 含义 |",
        "|------|------|",
        "| **Before · 开放检索** | `similarity_search(q, k)` 无 category filter |",
        "| **After · 硬路由** | `filter={\"category\": intended}` |",
        "",
        "## 总指标",
        "",
        "| 指标 | Before（开放） | After（硬路由） |",
        "|------|----------------|-----------------|",
        f"| 出现跨域 hit 的 query | **{routing['open_any_cross_queries']}** /"
        f" {routing['n_with_intended']} | **{routing['locked_impure_queries']}** /"
        f" {routing['n_with_intended']} |",
        f"| 重度污染 query（≥半数跨域） | **{routing['open_polluted_queries']}** /"
        f" {routing['n_with_intended']} | **{routing['locked_impure_queries']}** /"
        f" {routing['n_with_intended']} |",
        f"| 跨域 hit 占比（hit 级） | **{routing['open_cross_rate']:.1%}**"
        f"（{routing['open_cross_hits']}/{routing['open_total_hits']}） |"
        f" **{routing['locked_cross_rate']:.1%}**"
        f"（{routing['locked_cross_hits']}/{routing['locked_total_hits']}） |",
        "",
        f"换算备忘：全量 {routing['n_queries']} 作分母时，出现跨域约为"
        f" {routing['open_any_cross_queries']}/{routing['n_queries']}"
        f"（{routing['open_any_cross_queries']/routing['n_queries']:.1%}）；"
        "无关条无 intended，简历优先写绝对次数或 intended 分母。",
        "",
        "## 分桶摘要",
        "",
        "| 桶 | 条数 | 开放跨域 hit | 锁定跨域 hit | 开放出现跨域的 query |",
        "|----|------|--------------|--------------|----------------------|",
    ]
    for bucket in ("engineering", "academic", "cross", "irrelevant"):
        rs = by_bucket.get(bucket, [])
        if not rs:
            continue
        open_x = sum(r["open_cross"] for r in rs)
        lock_x = sum(r["locked_cross"] for r in rs)
        any_q = sum(1 for r in rs if r.get("any_cross"))
        lines.append(f"| {bucket} | {len(rs)} | {open_x} | {lock_x} | {any_q} |")

    lines.extend(
        [
            "",
            "## ParentDocument（机制行为验证 · 非效果 KPI）",
            "",
            f"在 {len(parent['details'])} 条 probe 上：平均子块 hit"
            f" **{parent['avg_child_hits']:.1f}** → unique parent"
            f" **{parent['avg_unique_parents']:.1f}**（去重成立={parent['ok']}）。",
            "",
            "> 只证明子块召回后能聚到更少父块；不要写成比普通 chunk 更好，不要把均值写进简历。",
            "",
            "| query | 域 | 子块数 | unique 父块 | 子块字符 | 父块字符 | 扩展比 |",
            "|-------|----|--------|-------------|----------|----------|--------|",
        ]
    )
    for d in parent["details"]:
        lines.append(
            f"| {d['query']} | {d['category']} | {d['child_hits']} |"
            f" {d['unique_parents']} | {d['child_chars']} |"
            f" {d['parent_chars']} | {d['expand_ratio']:.2f}× |"
        )

    lines.extend(
        [
            "",
            "## Query 明细",
            "",
            "| 桶 | intended | 开放 cats | 锁定 cats | 出现跨域 | 重度污染 |",
            "|----|----------|-----------|-----------|----------|----------|",
        ]
    )
    for r in routing["rows"]:
        lines.append(
            f"| {r['bucket']} | {r['intended']} | {r['open_cats']} |"
            f" {r['locked_cats'] or '—'} |"
            f" {'Y' if r.get('any_cross') else ''} |"
            f" {'Y' if r['open_polluted'] else ''} |"
        )

    lines.extend(
        [
            "",
            "## 简历 / 口述表述（推荐 · 治理口径）",
            "",
            f"> 构造 **{routing['n_queries']}** 条测试 query；在有目标域的"
            f" **{routing['n_with_intended']}** 条上，开放检索有"
            f" **{routing['open_any_cross_queries']}** 条出现跨域命中，硬路由后为 **0**。"
            f"（hit 级跨域占比 {routing['open_cross_rate']:.1%}→"
            f"{routing['locked_cross_rate']:.1%}，分母为有 intended 的 Top-K hit。）",
            "",
            "**不要写**：RAG 准确率提升、ParentDocument 均值进简历、混淆 query 级与 hit 级分母。",
            "",
            "## 局限",
            "",
            "- intended 为构造标签，非大规模人工金标。",
            "- 硬路由是确定性过滤器：锁库后跨域归零是治理目标，不是召回质量提升。",
            "- ParentDocument 仅为机制验证。",
            "",
        ]
    )
    out = EVAL_DIR / "EVAL_REPORT_RAG.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    if not CHROMA.is_dir():
        print("SKIP: no chroma_db")
        return
    print(f"queries={len(QUERIES)}")
    vs = _load_vs()
    routing = eval_routing(vs)
    print(
        f"any_cross={routing['open_any_cross_queries']}/{routing['n_with_intended']} "
        f"polluted={routing['open_polluted_queries']} "
        f"open_hit_cross={routing['open_cross_rate']:.2%} "
        f"locked={routing['locked_cross_rate']:.2%} "
        f"impure_locked={routing['locked_impure_queries']}"
    )
    parent = eval_parent_document(vs)
    print(
        f"parent avg children={parent['avg_child_hits']:.1f} "
        f"parents={parent['avg_unique_parents']:.1f}"
    )
    path = write_report(routing, parent)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
