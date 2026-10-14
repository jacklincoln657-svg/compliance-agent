"""本轮 Job 的 Policy Context 状态（切片 A）。

search_knowledge_base 成功后写入候选 ctx；
Agent 之后用 policy_refs 做交付单元归属（切片 B）。

ctx ≠ 归属：此处只记录「本轮出现过哪些 Policy Context」。
"""
from __future__ import annotations

import threading
import time
from typing import Any

from policy_registry import resolve_spec_for_kb

_lock = threading.RLock()
_contexts: list[dict[str, Any]] = []
_rag_call_seq = 0


def clear_policy_contexts() -> None:
    """新 Job 开始 / Job 结束时清空。"""
    global _contexts, _rag_call_seq
    with _lock:
        _contexts = []
        _rag_call_seq = 0


def list_policy_contexts() -> list[dict[str, Any]]:
    with _lock:
        return [dict(c) for c in _contexts]


def get_policy_context(ctx_id: str) -> dict[str, Any] | None:
    with _lock:
        for c in _contexts:
            if c.get("ctx_id") == ctx_id:
                return dict(c)
    return None


def known_ctx_ids() -> set[str]:
    with _lock:
        return {str(c["ctx_id"]) for c in _contexts if c.get("ctx_id")}


def _next_rag_call_id() -> str:
    global _rag_call_seq
    with _lock:
        _rag_call_seq += 1
        return f"rag_{_rag_call_seq}"


def _next_ctx_id() -> str:
    with _lock:
        return f"ctx_{len(_contexts) + 1}"


def append_policy_contexts_for_categories(
    categories: list[str] | set[str] | tuple[str, ...],
    *,
    rag_call_id: str | None = None,
    query: str | None = None,
) -> list[dict[str, Any]]:
    """
    一次 RAG 调用命中的 category 集合 → 为每个已注册 KB 追加一条 ctx。
    未在 KB_RULES 中的分类忽略（不产生 Policy）。
    同一 rag_call 内相同 category 只建一条。
    """
    call_id = rag_call_id or _next_rag_call_id()
    created: list[dict[str, Any]] = []
    seen_cat: set[str] = set()
    now = time.time()

    for raw in categories:
        cat = (raw or "").strip()
        if not cat or cat in seen_cat:
            continue
        seen_cat.add(cat)
        spec_id = resolve_spec_for_kb(cat)
        if not spec_id:
            continue
        with _lock:
            ctx = {
                "ctx_id": _next_ctx_id(),
                "kb": cat,
                "spec_id": spec_id,
                "created_at": now,
                "rag_call_id": call_id,
            }
            if query:
                ctx["query"] = query[:200]
            _contexts.append(ctx)
            created.append(dict(ctx))
    return created


def begin_rag_call() -> str:
    """一次 search 开始时取 rag_call_id（便于同次调用挂多个 ctx）。"""
    return _next_rag_call_id()
