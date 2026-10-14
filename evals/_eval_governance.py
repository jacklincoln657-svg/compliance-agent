"""
项目一 · 其余能力点的可复现评测（不调真实 LLM API）

覆盖简历三条中除「硬校验」以外、能量化的部分：
  1) RAG 硬路由：有/无 category filter 的串库率（需已 build_knowledge）
  2) ParentDocument：子块命中后父块去重交付
  3) Tool 轨迹清洗：发给模型前的字符量下降
  4) 弹性摘要窗口：Max/Keep 下摘要调用次数（FakeLLM）
  5) COW 中断回滚：改旧文件 / 新建文件是否还原或删除
  6) 沙箱后端：探测 docker|local（能力项，非准确率）

用法：
  cd cyber_lobster_project
  python evals/_eval_governance.py

报告写入 evals/EVAL_REPORT_GOVERNANCE.md
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import json
import os
import shutil
import tempfile
from pathlib import Path

from _bootstrap import EVAL_DIR, ROOT
CHROMA = ROOT / "chroma_db"
PARENT_STORE = ROOT / "parent_store.json"

# 避免本机无网时 HuggingFace HEAD 重试拖死评测（本地已有模型缓存即可）
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


class FakeLLM:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, prompt: str):
        self.calls += 1

        class R:
            content = f"SUMMARY#{self.calls}"

        return R()


def section(title: str) -> None:
    print(f"\n=== {title} ===")


# ---------- 1) Context: tool strip ----------
def eval_tool_strip() -> dict:
    from context_manager import extract_visible_dialogue

    history = []
    for i in range(3):
        history.append({"role": "user", "content": f"用户指令{i}：" + ("详请" * 40)})
        history.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"name": "execute_python_code", "args": {"code": "x" * 2000}}],
            }
        )
        history.append(
            {
                "role": "tool",
                "content": "stdout:\n" + ("ERROR TRACEBACK LINE\n" * 80),
            }
        )
        history.append(
            {
                "role": "assistant",
                "content": f"本轮结论{i}：已完成。" + ("说明" * 20),
            }
        )

    raw_chars = sum(len(json.dumps(m, ensure_ascii=False)) for m in history)
    visible = extract_visible_dialogue(history)
    vis_chars = sum(len(json.dumps(m, ensure_ascii=False)) for m in visible)
    ratio = 1 - (vis_chars / raw_chars) if raw_chars else 0
    return {
        "name": "Tool轨迹清洗",
        "raw_chars": raw_chars,
        "visible_chars": vis_chars,
        "reduction": ratio,
        "visible_msgs": len(visible),
        "raw_msgs": len(history),
        "ok": vis_chars < raw_chars and len(visible) < len(history),
    }


# ---------- 2) Elastic window ----------
def eval_elastic_window() -> dict:
    from context_manager import build_model_context, KEEP_RECENT_USER_TURNS, MAX_RAW_USER_TURNS

    def hist(n: int) -> list:
        out = []
        for i in range(1, n + 1):
            out.append({"role": "user", "content": f"U{i} " + ("任务内容" * 30)})
            out.append({"role": "assistant", "content": f"A{i} " + ("回复" * 30)})
        return out

    llm = FakeLLM()
    # 静默：8 轮不摘要
    build_model_context(hist(MAX_RAW_USER_TURNS), "", llm, summary_covers=0)
    silent_calls = llm.calls
    # 第 9 轮触发
    _, _, covers = build_model_context(hist(MAX_RAW_USER_TURNS + 1), "", llm, summary_covers=0)
    trigger_calls = llm.calls
    # 再静默到 covers+MAX
    start = llm.calls
    c = covers
    s = "SUMMARY#1"
    for n in range(covers + KEEP_RECENT_USER_TURNS + 1, covers + MAX_RAW_USER_TURNS + 1):
        _, s, c = build_model_context(hist(n), s, llm, summary_covers=c)
    silent2 = llm.calls - start
    # 再次触发
    before = llm.calls
    build_model_context(hist(covers + MAX_RAW_USER_TURNS + 1), s, llm, summary_covers=c)
    second_trigger = llm.calls - before

    # 对照：若用旧策略「过 Keep 就每轮摘要」——从第 5 轮起每轮都调
    naive_calls = max(0, (covers + MAX_RAW_USER_TURNS + 1) - KEEP_RECENT_USER_TURNS)
    # 粗估：用户轮到 14 时 naive ≈ 10 次；弹性 ≈ 2 次
    elastic_calls = llm.calls

    return {
        "name": "弹性摘要窗口",
        "max": MAX_RAW_USER_TURNS,
        "keep": KEEP_RECENT_USER_TURNS,
        "silent_at_8": silent_calls == 0,
        "trigger_at_9": trigger_calls == 1,
        "silent_between": silent2 == 0,
        "second_trigger": second_trigger == 1,
        "elastic_total_calls_to_14": elastic_calls,
        "naive_estimate_every_turn_after_keep": naive_calls,
        "ok": silent_calls == 0 and trigger_calls == 1 and silent2 == 0 and second_trigger == 1,
    }


# ---------- 3) COW ----------
def eval_cow() -> dict:
    from cow_backup import CowSession, build_sandbox_cow_preamble
    import subprocess
    import sys

    td = Path(tempfile.mkdtemp(prefix="eval_cow_"))
    try:
        old = td / "keep_me.txt"
        old.write_text("ORIGINAL", encoding="utf-8")
        session = CowSession(str(td))
        preamble = build_sandbox_cow_preamble(
            session.workspace_dir, session.backup_root, session.initial_files
        )
        script = td / "temp_sandbox_script.py"
        body = (
            preamble
            + "\n"
            + "open('keep_me.txt','w',encoding='utf-8').write('DIRTY')\n"
            + "open('brand_new.txt','w',encoding='utf-8').write('NEW')\n"
        )
        script.write_text(body, encoding="utf-8")
        subprocess.run([sys.executable, str(script)], cwd=str(td), check=True, capture_output=True)
        assert (td / "keep_me.txt").read_text(encoding="utf-8") == "DIRTY"
        assert (td / "brand_new.txt").is_file()

        restored, deleted = session.rollback()
        ok_restore = (td / "keep_me.txt").read_text(encoding="utf-8") == "ORIGINAL"
        ok_deleted = not (td / "brand_new.txt").exists()
        session.cleanup()
        return {
            "name": "COW中断回滚",
            "restored": restored,
            "deleted_new": deleted,
            "old_file_restored": ok_restore,
            "new_file_removed": ok_deleted,
            "ok": ok_restore and ok_deleted and restored >= 1 and deleted >= 1,
        }
    finally:
        shutil.rmtree(td, ignore_errors=True)


# ---------- 4) RAG hard routing ----------
def eval_rag_routing() -> dict:
    if not CHROMA.is_dir() or not PARENT_STORE.is_file():
        return {
            "name": "RAG硬路由",
            "skipped": True,
            "reason": "未找到 chroma_db/parent_store.json，请先 python build_knowledge.py",
            "ok": None,
        }

    vs = None
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import Chroma

        emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vs = Chroma(persist_directory=str(CHROMA), embedding_function=emb)
    except Exception:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_community.vectorstores import Chroma

            emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vs = Chroma(persist_directory=str(CHROMA), embedding_function=emb)
        except Exception as e2:
            return {
                "name": "RAG硬路由",
                "skipped": True,
                "reason": f"向量依赖不可用（请在项目 venv 中跑）: {e2}",
                "ok": None,
            }

    query = "毕业设计任务书 开题报告 进度安排 研究内容"
    k = 6
    open_hits = vs.similarity_search(query, k=k)
    locked_hits = vs.similarity_search(query, k=k, filter={"category": "engineering"})

    def cats(hits):
        return [((h.metadata or {}).get("category") or "") for h in hits]

    open_cats = cats(open_hits)
    locked_cats = cats(locked_hits)
    open_cross = sum(1 for c in open_cats if c and c != "engineering")
    locked_cross = sum(1 for c in locked_cats if c and c != "engineering")

    parent_ids = []
    for h in locked_hits or open_hits:
        pid = (h.metadata or {}).get("parent_id")
        if pid:
            parent_ids.append(pid)
    unique_parents = len(set(parent_ids))

    return {
        "name": "RAG硬路由",
        "skipped": False,
        "query": query,
        "open_categories": open_cats,
        "locked_engineering_categories": locked_cats,
        "open_non_engineering": open_cross,
        "locked_non_engineering": locked_cross,
        "unique_parent_ids": unique_parents,
        "child_hits": len(locked_hits or open_hits),
        "ok": locked_cross == 0,
    }


# ---------- 5) Sandbox backend capability ----------
def eval_sandbox_backend() -> dict:
    """不 import tools_sandbox（避免当前解释器缺 langchain）；直接探测 Docker。"""
    import shutil
    import subprocess

    cli = shutil.which("docker")
    docker_ok = False
    if cli:
        try:
            r = subprocess.run(
                [cli, "info"],
                capture_output=True,
                text=True,
                timeout=8,
            )
            docker_ok = r.returncode == 0
        except Exception:
            docker_ok = False

    # 与 tools_sandbox.resolve 语义对齐的简化版
    mode = (os.getenv("SANDBOX_BACKEND") or "auto").strip().lower()
    if mode == "local":
        backend = "local"
    elif mode == "docker":
        backend = "docker" if docker_ok else "local"
    else:
        backend = "docker" if docker_ok else "local"

    return {
        "name": "沙箱后端探测",
        "backend": backend,
        "docker_available": docker_ok,
        "label": f"{backend}",
        "ok": backend in ("docker", "local"),
        "note": "此项为能力探测，不是 before/after 准确率；Docker 可用时简历可写「实测优先容器」。",
    }


def main() -> None:
    results = []
    section("1 Tool strip")
    r1 = eval_tool_strip()
    results.append(r1)
    print(r1)

    section("2 Elastic window")
    r2 = eval_elastic_window()
    results.append(r2)
    print(r2)

    section("3 COW")
    r3 = eval_cow()
    results.append(r3)
    print(r3)

    section("4 RAG routing")
    r4 = eval_rag_routing()
    results.append(r4)
    print(r4)

    section("5 Sandbox backend")
    r5 = eval_sandbox_backend()
    results.append(r5)
    print(r5)

    # Markdown report
    lines = [
        "# 本地数据分析 Agent · 治理/RAG/沙箱评测报告",
        "",
        "> 脚本：`_eval_governance.py`（不调用收费 LLM API）",
        "> 硬校验 before/after 见 `EVAL_REPORT.md` / `_eval_before_after.py`",
        "",
        "## 总览",
        "",
        "| 能力点 | 结果 | 说明 |",
        "|--------|------|------|",
    ]

    def flag(ok):
        if ok is True:
            return "PASS"
        if ok is False:
            return "FAIL"
        return "SKIP"

    lines.append(
        f"| Tool轨迹清洗 | {flag(r1['ok'])} | 字符量 {r1['raw_chars']}→{r1['visible_chars']}，降幅 **{r1['reduction']:.0%}** |"
    )
    lines.append(
        f"| 弹性摘要窗口 | {flag(r2['ok'])} | Max={r2['max']}/Keep={r2['keep']}；至约14轮摘要调用 **{r2['elastic_total_calls_to_14']}** 次"
        f"（若过Keep每轮摘要约 **{r2['naive_estimate_every_turn_after_keep']}** 次） |"
    )
    lines.append(
        f"| COW回滚 | {flag(r3['ok'])} | 恢复旧文件={r3['old_file_restored']}，删除新建={r3['new_file_removed']} "
        f"(restored={r3['restored']}, deleted={r3['deleted_new']}) |"
    )
    if r4.get("skipped"):
        lines.append(f"| RAG硬路由 | SKIP | {r4.get('reason')} |")
    else:
        lines.append(
            f"| RAG硬路由 | {flag(r4['ok'])} | 无filter串到非engineering **{r4['open_non_engineering']}/{len(r4['open_categories'])}**；"
            f"锁engineering后串库 **{r4['locked_non_engineering']}**；unique parent_id={r4['unique_parent_ids']} |"
        )
    lines.append(
        f"| 沙箱后端 | {flag(r5['ok'])} | 当前 **{r5['backend']}**（docker可用={r5['docker_available']}）—"
        f"{r5['note']} |"
    )

    lines.extend(
        [
            "",
            "## 与简历三条 bullet 的对应",
            "",
            "| 简历表述 | 本评测能量化的部分 | 暂不能装成「准确率%」的部分 |",
            "|----------|-------------------|------------------------------|",
            "| 可验证交付与合规硬闸 | → 见 `EVAL_REPORT.md`（软检查 vs 硬闸） | Checklist/脚本自检仍是 Prompt 软约束 |",
            "| Policy RAG + 硬路由 | 有/无 category filter 串库次数；parent_id 去重规模 | 「规范约束了 Agent 行为」需端到端 LLM 评测 |",
            "| 受控执行与状态治理 | COW 回滚成败；tool清洗降幅；弹性摘要调用次数；Docker是否可用 | Docker vs local 的逃逸对比需额外攻击用例 |",
            "",
            "## 简历可引用的诚实句子（示例）",
            "",
            f"1. Tool轨迹清洗：合成多轮历史字符量下降约 **{r1['reduction']:.0%}**。",
            f"2. 弹性窗口：模拟至约14用户轮，摘要 LLM 调用 **{r2['elastic_total_calls_to_14']}** 次"
            f"（对照「过Keep每轮摘要」估约 **{r2['naive_estimate_every_turn_after_keep']}** 次）。",
            f"3. COW：注入改旧+新建后 rollback，旧内容恢复且新建删除（本机合成用例 **{'通过' if r3['ok'] else '失败'}**）。",
        ]
    )
    if not r4.get("skipped"):
        lines.append(
            f"4. 知识硬路由：查询偏任务书语义时，无过滤串出 engineering 外类别 "
            f"**{r4['open_non_engineering']}** 次，锁定 engineering 后串库 **{r4['locked_non_engineering']}** 次。"
        )
    lines.append(
        f"5. 沙箱：本机探测后端为 **{r5['backend']}**"
        + ("（Docker 可用）" if r5["docker_available"] else "（当前回退 local，勿写已强制容器隔离）。")
    )
    lines.extend(
        [
            "",
            "## 局限",
            "",
            "- 未跑真实 DeepSeek 多轮 Agent，故无「端到端任务成功率」。",
            "- RAG 数字依赖当前 `chroma_db` 语料；换库需重跑。",
            "- Checklist 层无法用本脚本单独量化。",
            "",
        ]
    )

    out = EVAL_DIR / "EVAL_REPORT_GOVERNANCE.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {out}")
    failed = [r for r in results if r.get("ok") is False]
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
