"""切片 A 冒烟：KB_RULES + Policy Context Job State（不改 Prompt/UI/finalize）。

运行（在项目根）: python smokes/_smoke_policy_ctx.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import spec_validator as sv
from job_policy_state import (
    append_policy_contexts_for_categories,
    begin_rag_call,
    clear_policy_contexts,
    get_policy_context,
    known_ctx_ids,
    list_policy_contexts,
)
from policy_registry import KB_RULES, list_kb_policy_map, resolve_spec_for_kb


def test_kb_rules_map_to_existing_specs() -> None:
    mapping = list_kb_policy_map()
    assert mapping["engineering"] == "export_desensitize_v3"
    assert mapping["academic"] == "thesis_outline_v1"
    for sid in mapping.values():
        assert sid in sv.RULES, f"{sid} not in RULES"
        # require_check 只存在 RULES，不在 KB_RULES
        assert "require_check" not in KB_RULES[next(
            c for c, s in mapping.items() if s == sid
        )]
    assert resolve_spec_for_kb("unknown_lib") is None
    assert resolve_spec_for_kb("__all__") is None
    print("[OK] KB_RULES → RULES spec_id; no require_check in KB_RULES")


def test_append_single_and_multi_category() -> None:
    clear_policy_contexts()
    rag1 = begin_rag_call()
    created = append_policy_contexts_for_categories(
        ["engineering"], rag_call_id=rag1, query="phone mask"
    )
    assert len(created) == 1, created
    assert created[0]["ctx_id"] == "ctx_1"
    assert created[0]["kb"] == "engineering"
    assert created[0]["spec_id"] == "export_desensitize_v3"
    assert created[0]["rag_call_id"] == rag1
    assert "created_at" in created[0]

    # 同一次调用命中两个库（模拟 __all__）
    rag2 = begin_rag_call()
    created2 = append_policy_contexts_for_categories(
        ["engineering", "academic", "engineering"],  # 去重
        rag_call_id=rag2,
    )
    assert len(created2) == 2, created2
    specs = {c["spec_id"] for c in created2}
    assert specs == {"export_desensitize_v3", "thesis_outline_v1"}
    assert all(c["rag_call_id"] == rag2 for c in created2)

    all_ctx = list_policy_contexts()
    assert len(all_ctx) == 3  # ctx_1 + ctx_2 + ctx_3
    assert known_ctx_ids() == {"ctx_1", "ctx_2", "ctx_3"}
    assert get_policy_context("ctx_2") is not None
    assert get_policy_context("ctx_999") is None
    print("[OK] single + multi-category ctx; dedupe within rag_call")


def test_unregistered_category_no_ctx() -> None:
    clear_policy_contexts()
    created = append_policy_contexts_for_categories(["通用", "not_a_kb"])
    assert created == []
    assert list_policy_contexts() == []
    print("[OK] unregistered category → no Policy Context")


def test_clear_resets_ids() -> None:
    clear_policy_contexts()
    append_policy_contexts_for_categories(["academic"])
    assert list_policy_contexts()[0]["ctx_id"] == "ctx_1"
    clear_policy_contexts()
    assert list_policy_contexts() == []
    created = append_policy_contexts_for_categories(["academic"])
    assert created[0]["ctx_id"] == "ctx_1"
    print("[OK] clear_policy_contexts resets counters")


def test_ctx_is_candidate_not_attribution() -> None:
    """文档化约束：list 只含候选，不含 file 绑定。"""
    clear_policy_contexts()
    append_policy_contexts_for_categories(["engineering", "academic"])
    for c in list_policy_contexts():
        assert "files" not in c
        assert "policy_refs" not in c
        assert set(c.keys()) >= {
            "ctx_id",
            "kb",
            "spec_id",
            "created_at",
            "rag_call_id",
        }
    print("[OK] ctx fields = candidate metadata only (no file attribution)")


def main() -> None:
    print("=== smoke: policy context (slice A) ===")
    test_kb_rules_map_to_existing_specs()
    test_append_single_and_multi_category()
    test_unregistered_category_no_ctx()
    test_clear_resets_ids()
    test_ctx_is_candidate_not_attribution()
    print("=== ALL SLICE-A SMOKE PASSED ===")


if __name__ == "__main__":
    main()
