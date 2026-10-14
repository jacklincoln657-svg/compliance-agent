"""policy_refs 补充冒烟：与 _smoke_freeze_scenarios 不重叠的路径。

结构场景见 smokes/_smoke_freeze_scenarios.py；本文件覆盖：
- REGISTERED_NO_POLICY 对用户文案
- 同单元多 refs 且缺产物
- legacy spec_id 兼容警告
- runtime hint / 修复反馈里的 ctx 快照

运行（在项目根）: python smokes/_smoke_policy_refs.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import json
import shutil
import tempfile
from pathlib import Path

import spec_validator as sv
from job_policy_state import append_policy_contexts_for_categories, clear_policy_contexts


def _write_export_pair(root: Path, sub: str = "export_batch_v3") -> None:
    import csv

    d = root / sub
    d.mkdir(parents=True, exist_ok=True)
    with (d / "clean_users_data.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Name", "Phone", "Score"])
        w.writerow(["Alice", "138****5678", "58.90"])
    (d / "audit_log.json").write_text(
        json.dumps({"total_rows": 1, "missing_scores_filled": 0}, ensure_ascii=False),
        encoding="utf-8",
    )


def test_empty_refs_user_message() -> None:
    clear_policy_contexts()
    td = Path(tempfile.mkdtemp())
    try:
        (td / "chart.png").write_bytes(b"x")
        (td / sv.MANIFEST_NAME).write_text(
            json.dumps(
                {
                    "deliveries": [
                        {"files": ["chart.png"], "policy_refs": []},
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        r = sv.finalize_job_delivery(str(td), new_files=["chart.png"])
        assert r.get("ok") and r.get("status") == "REGISTERED_NO_POLICY", r
        um = sv.format_user_delivery_message(r)
        assert "NO_POLICY" in um
        assert "未" in um and "证明" in um
        print("[OK] refs=[] → REGISTERED_NO_POLICY + user copy")
    finally:
        shutil.rmtree(td, ignore_errors=True)


def test_multi_refs_any_fail() -> None:
    """同一交付单元绑两个 ctx：export 过、但缺 thesis 文件 → 单元 FAIL。"""
    clear_policy_contexts()
    c1 = append_policy_contexts_for_categories(["engineering"])[0]
    c2 = append_policy_contexts_for_categories(["academic"])[0]
    td = Path(tempfile.mkdtemp())
    try:
        _write_export_pair(td)
        files = [
            "export_batch_v3/clean_users_data.csv",
            "export_batch_v3/audit_log.json",
        ]
        (td / sv.MANIFEST_NAME).write_text(
            json.dumps(
                {
                    "deliveries": [
                        {
                            "files": files,
                            "policy_refs": [c1["ctx_id"], c2["ctx_id"]],
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        r = sv.finalize_job_delivery(str(td), new_files=files)
        assert not r.get("ok"), r
        blob = " ".join(r.get("errors") or [])
        assert (
            "thesis" in blob.lower()
            or "篇幅" in blob
            or "主题" in blob
            or "开题" in blob
            or "md" in blob.lower()
            or "不存在" in blob
            or "缺少" in blob
        )
        print("[OK] multi refs: any validator FAIL → unit FAIL")
    finally:
        shutil.rmtree(td, ignore_errors=True)
        clear_policy_contexts()


def test_legacy_spec_id_path_still_works() -> None:
    """无 policy_refs 时按声明 spec 验收，并打迁移兼容警告。"""
    clear_policy_contexts()
    td = Path(tempfile.mkdtemp())
    try:
        (td / "out.txt").write_text("x", encoding="utf-8")
        (td / sv.MANIFEST_NAME).write_text(
            json.dumps({"spec_id": "normal", "files": ["out.txt"]}, ensure_ascii=False),
            encoding="utf-8",
        )
        r = sv.finalize_job_delivery(str(td), new_files=["out.txt"])
        assert r.get("ok") and r.get("status") == "REGISTERED", r
        assert r.get("legacy_compat") is True
        assert any("迁移兼容" in w for w in (r.get("warnings") or []))
        print("[OK] legacy spec_id path (compat warning)")
    finally:
        shutil.rmtree(td, ignore_errors=True)


def test_runtime_hint_policy_refs() -> None:
    clear_policy_contexts()
    h0 = sv.build_compliance_runtime_hint()
    assert "policy_refs" in h0
    assert "Policy Context" in h0 or "ctx" in h0.lower()
    assert "尚无" in h0 or "[]" in h0

    created = append_policy_contexts_for_categories(["engineering", "academic"])
    h1 = sv.build_compliance_runtime_hint()
    assert created[0]["ctx_id"] in h1
    assert created[1]["ctx_id"] in h1
    assert "export_desensitize_v3" in h1
    snap = sv.format_policy_contexts_hint()
    assert "ctx_1" in snap
    assert "引用" in snap or "attribution" in snap.lower()
    print("[OK] runtime hint lists policy_refs + live ctx snapshot")
    clear_policy_contexts()


def test_feedback_includes_ctx_snapshot() -> None:
    clear_policy_contexts()
    append_policy_contexts_for_categories(["engineering"])
    r = {
        "ok": False,
        "status": "FAIL",
        "provenance_mode": True,
        "errors": ["未知或本轮不存在的 policy_refs: `ctx_nope`"],
    }
    fb = sv.format_validation_feedback(r)
    assert "policy_refs" in fb
    assert "ctx_1" in fb
    assert "deliveries" in fb
    print("[OK] validation feedback includes ctx snapshot + policy_refs fix guidance")
    clear_policy_contexts()


def main() -> None:
    print("=== smoke: policy_refs supplements ===")
    test_empty_refs_user_message()
    test_multi_refs_any_fail()
    test_legacy_spec_id_path_still_works()
    test_runtime_hint_policy_refs()
    test_feedback_includes_ctx_snapshot()
    print("=== ALL POLICY_REFS SUPPLEMENTS PASSED ===")


if __name__ == "__main__":
    main()
