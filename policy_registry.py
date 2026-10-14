"""知识库分类 → Policy（spec_id）注册表。

KB_RULES 只回答「用了这个 KB 获得哪个 Policy」。
是否硬验看 RULES[spec_id].require_check，此处不重复存 require_check。
"""
from __future__ import annotations

from typing import Any

# category（knowledge_docs 子目录名）→ 对应的 RULES spec_id
KB_RULES: dict[str, dict[str, Any]] = {
    "engineering": {
        "spec_id": "export_desensitize_v3",
        "label": "工程/出仓脱敏知识库",
    },
    "academic": {
        "spec_id": "thesis_outline_v1",
        "label": "学术/开题任务书知识库",
    },
}


def resolve_spec_for_kb(category: str | None) -> str | None:
    """有注册则返回 spec_id；未注册的 KB 不产生 Policy Context。"""
    cat = (category or "").strip()
    if not cat or cat in ("__all__", "__none__", "通用"):
        return None
    cfg = KB_RULES.get(cat)
    if not cfg:
        return None
    sid = cfg.get("spec_id")
    return str(sid).strip() if sid else None


def list_kb_policy_map() -> dict[str, str]:
    """category → spec_id（便于调试/冒烟）。"""
    return {
        cat: str(cfg.get("spec_id"))
        for cat, cfg in KB_RULES.items()
        if cfg.get("spec_id")
    }
