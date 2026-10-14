"""冒烟：交付清单单/多交付、RULES、runtime hint。

运行（在项目根）: python smokes/_smoke_compliance.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import csv
import json
import shutil
import tempfile
from pathlib import Path

import spec_validator as sv


def _fail(msg: str) -> None:
    raise AssertionError(msg)


def _write_export_pair(root: Path, sub: str = "export_batch_v3") -> None:
    d = root / sub
    d.mkdir(parents=True, exist_ok=True)
    csv_path = d / "clean_users_data.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Name", "Phone", "Score"])
        w.writerow(["Alice", "138****5678", "58.90"])
        w.writerow(["Bob", "139****1234", "70.00"])
    (d / "audit_log.json").write_text(
        json.dumps({"total_rows": 2, "missing_scores_filled": 0}, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_thesis(root: Path, rel: str = "reports/开题报告.md", *, good: bool = True) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if good:
        body = (
            "一、毕业设计原始依据与背景：智慧医疗与信息孤岛、多模态数据。\n"
            "二、主要内容与研究内容：文本结构化、影像特征提取、跨模态检索。\n"
            "三、基本要求：提交可运行代码、答辩演示、隐私保护。\n"
            "四、进度安排：完成开题报告与任务书，后续中期检查。\n"
            + ("补充说明。" * 40)
        )
    else:
        body = "太短了"
    path.write_text(body, encoding="utf-8")


def test_no_manifest_skipped() -> None:
    td = Path(tempfile.mkdtemp())
    try:
        r = sv.validate_compliance(str(td))
        assert r.get("skipped") is True and r.get("ok") is True, r
        print("[OK] no manifest → skipped")
    finally:
        shutil.rmtree(td, ignore_errors=True)


def test_legacy_single_ok() -> None:
    td = Path(tempfile.mkdtemp())
    try:
        _write_export_pair(td)
        (td / sv.MANIFEST_NAME).write_text(
            json.dumps(
                {
                    "spec_id": "export_desensitize_v3",
                    "files": [
                        "export_batch_v3/clean_users_data.csv",
                        "export_batch_v3/audit_log.json",
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        raw = sv.load_compliance_manifest(str(td))
        assert raw and len(raw["deliveries"]) == 1, raw
        r = sv.validate_compliance(str(td))
        assert r.get("ok") is True and not r.get("skipped"), r
        assert r.get("spec_id") == "export_desensitize_v3", r
        print("[OK] legacy single export pass")
    finally:
        shutil.rmtree(td, ignore_errors=True)


def test_multi_both_ok() -> None:
    td = Path(tempfile.mkdtemp())
    try:
        _write_export_pair(td)
        _write_thesis(td, good=True)
        (td / sv.MANIFEST_NAME).write_text(
            json.dumps(
                {
                    "deliveries": [
                        {
                            "spec_id": "export_desensitize_v3",
                            "files": [
                                "export_batch_v3/clean_users_data.csv",
                                "export_batch_v3/audit_log.json",
                            ],
                        },
                        {
                            "spec_id": "thesis_outline_v1",
                            "files": ["reports/开题报告.md"],
                        },
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        r = sv.validate_compliance(str(td))
        assert r.get("ok") is True, r
        assert len(r.get("results") or []) == 2, r
        print("[OK] multi deliveries both pass")
    finally:
        shutil.rmtree(td, ignore_errors=True)


def test_multi_partial_fail_prefix() -> None:
    td = Path(tempfile.mkdtemp())
    try:
        _write_export_pair(td)
        _write_thesis(td, good=False)
        (td / sv.MANIFEST_NAME).write_text(
            json.dumps(
                {
                    "deliveries": [
                        {
                            "spec_id": "export_desensitize_v3",
                            "files": [
                                "export_batch_v3/clean_users_data.csv",
                                "export_batch_v3/audit_log.json",
                            ],
                        },
                        {
                            "spec_id": "thesis_outline_v1",
                            "files": ["reports/开题报告.md"],
                        },
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        r = sv.validate_compliance(str(td))
        assert r.get("ok") is False, r
        errs = r.get("errors") or []
        assert any(e.startswith("[thesis_outline_v1]") for e in errs), errs
        assert not any(
            e.startswith("[export_desensitize_v3]") for e in errs
        ), f"export should pass: {errs}"
        fb = sv.format_validation_feedback(r)
        assert "[thesis_outline_v1]" in fb or "thesis_outline_v1" in fb
        print("[OK] multi partial fail + [spec_id] prefix")
    finally:
        shutil.rmtree(td, ignore_errors=True)


def test_unknown_spec() -> None:
    td = Path(tempfile.mkdtemp())
    try:
        (td / "x.txt").write_text("hi", encoding="utf-8")
        (td / sv.MANIFEST_NAME).write_text(
            json.dumps({"spec_id": "no_such_spec", "files": ["x.txt"]}),
            encoding="utf-8",
        )
        r = sv.validate_compliance(str(td))
        assert r.get("ok") is False
        assert any("未知 spec_id" in e for e in (r.get("errors") or [])), r
        print("[OK] unknown spec_id rejected")
    finally:
        shutil.rmtree(td, ignore_errors=True)


def test_cross_file_not_validated_as_export() -> None:
    """开题文件误塞进出仓清单：按 basename 找不到 csv → 应失败，而非用脱敏规则扫 md。"""
    td = Path(tempfile.mkdtemp())
    try:
        _write_thesis(td, good=True)
        (td / sv.MANIFEST_NAME).write_text(
            json.dumps(
                {
                    "spec_id": "export_desensitize_v3",
                    "files": ["reports/开题报告.md"],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        r = sv.validate_compliance(str(td))
        assert r.get("ok") is False, r
        blob = " ".join(r.get("errors") or [])
        assert "clean_users_data.csv" in blob or "缺少" in blob, r
        print("[OK] thesis path under export spec → fail (missing export files)")
    finally:
        shutil.rmtree(td, ignore_errors=True)


def test_runtime_hint() -> None:
    h = sv.build_compliance_runtime_hint()
    for sid in sv.RULES:
        assert sid in h, sid
    assert sv.MANIFEST_NAME in h
    assert "policy_refs" in h
    assert "export_desensitize_v3" in h
    print("[OK] build_compliance_runtime_hint")


def test_clear_manifest() -> None:
    td = Path(tempfile.mkdtemp())
    try:
        p = td / sv.MANIFEST_NAME
        p.write_text("{}", encoding="utf-8")
        sv.clear_compliance_manifest(str(td))
        assert not p.exists()
        print("[OK] clear_compliance_manifest")
    finally:
        shutil.rmtree(td, ignore_errors=True)


def test_finalize_matrix() -> None:
    """有新文件必须清单；legacy spec_id 按声明验收。"""
    td = Path(tempfile.mkdtemp())
    try:
        r = sv.finalize_job_delivery(str(td))
        assert r.get("status") == "NO_FORMAL_DELIVERY" and r.get("ok"), r
        print("[OK] finalize: no files no manifest → NO_FORMAL_DELIVERY")
    finally:
        shutil.rmtree(td, ignore_errors=True)

    td = Path(tempfile.mkdtemp())
    try:
        (td / "out.txt").write_text("x", encoding="utf-8")
        (td / sv.MANIFEST_NAME).write_text(
            json.dumps({"spec_id": "normal", "files": ["out.txt"]}, ensure_ascii=False),
            encoding="utf-8",
        )
        r = sv.finalize_job_delivery(str(td), new_files=["out.txt"])
        assert r.get("ok") and r.get("status") == "REGISTERED", r
        print("[OK] finalize: legacy normal → REGISTERED")
    finally:
        shutil.rmtree(td, ignore_errors=True)

    td = Path(tempfile.mkdtemp())
    try:
        _write_export_pair(td)
        (td / sv.MANIFEST_NAME).write_text(
            json.dumps(
                {
                    "spec_id": "export_desensitize_v3",
                    "files": [
                        "export_batch_v3/clean_users_data.csv",
                        "export_batch_v3/audit_log.json",
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        r = sv.finalize_job_delivery(str(td))
        assert r.get("ok") and r.get("status") == "PASS", r
        print("[OK] finalize: legacy export spec → PASS")
    finally:
        shutil.rmtree(td, ignore_errors=True)

    td = Path(tempfile.mkdtemp())
    try:
        _write_export_pair(td)
        r = sv.finalize_job_delivery(str(td))
        assert r.get("status") == "NO_FORMAL_DELIVERY" and r.get("ok"), r
        print("[OK] finalize: files on disk, no new_files arg → NO_FORMAL")
    finally:
        shutil.rmtree(td, ignore_errors=True)


def test_phase2_force_manifest() -> None:
    """本轮新文件无清单 → MISSING_MANIFEST；有清单但漏列 → FAIL。"""
    td = Path(tempfile.mkdtemp())
    try:
        news = ["notes/draft.txt", "out.csv"]
        r = sv.finalize_job_delivery(str(td), new_files=news)
        assert not r.get("ok") and r.get("status") == "MISSING_MANIFEST", r
        um = sv.format_user_delivery_message(r)
        assert "交付清单" in um and "瞒报" in um
        print("[OK] phase2: new_files + no manifest → MISSING_MANIFEST")
    finally:
        shutil.rmtree(td, ignore_errors=True)

    td = Path(tempfile.mkdtemp())
    try:
        (td / "a.txt").write_text("1", encoding="utf-8")
        (td / "b.txt").write_text("2", encoding="utf-8")
        (td / sv.MANIFEST_NAME).write_text(
            json.dumps({"spec_id": "normal", "files": ["a.txt"]}, ensure_ascii=False),
            encoding="utf-8",
        )
        r = sv.finalize_job_delivery(str(td), new_files=["a.txt", "b.txt"])
        assert not r.get("ok") and r.get("status") == "FAIL", r
        blob = " ".join(r.get("errors") or [])
        assert "未列入清单" in blob and "b.txt" in blob, r
        kind = sv._classify_delivery_errors(r)
        assert kind == "files_not_listed", kind
        print("[OK] phase2: manifest missing file in files → FAIL")
    finally:
        shutil.rmtree(td, ignore_errors=True)

    td = Path(tempfile.mkdtemp())
    try:
        (td / "a.txt").write_text("1", encoding="utf-8")
        (td / "legacy_ok.txt").write_text("old", encoding="utf-8")
        (td / sv.MANIFEST_NAME).write_text(
            json.dumps(
                {"spec_id": "normal", "files": ["a.txt", "legacy_ok.txt"]},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        r = sv.finalize_job_delivery(str(td), new_files=["a.txt"])
        assert r.get("ok") and r.get("status") == "REGISTERED", r
        print("[OK] phase2: files covers new (+ optional extras) → REGISTERED")
    finally:
        shutil.rmtree(td, ignore_errors=True)


def test_cow_list_new_files() -> None:
    from cow_backup import CowSession

    td = Path(tempfile.mkdtemp())
    try:
        (td / "old.txt").write_text("old", encoding="utf-8")
        cow = CowSession(str(td))
        assert cow.list_new_files() == []
        (td / "new_dir").mkdir()
        (td / "new_dir" / "fresh.txt").write_text("n", encoding="utf-8")
        news = cow.list_new_files()
        assert news == ["new_dir/fresh.txt"], news
        cow.cleanup()
        print("[OK] CowSession.list_new_files")
    finally:
        shutil.rmtree(td, ignore_errors=True)


def test_prompt_mentions_rules() -> None:
    """可选：有 langchain 环境时检查 system_prompt 与 RULES 同步。"""
    try:
        from agent_core import system_prompt
    except Exception as e:
        print(f"[SKIP] agent_core import ({e})")
        return
    for sid in sv.RULES:
        assert sid in system_prompt, f"missing {sid} in system_prompt"
    assert "policy_refs" in system_prompt
    assert "Policy Context" in system_prompt or "policy_refs" in system_prompt
    print("[OK] agent_core system_prompt lists RULES + policy_refs")


def main() -> None:
    print("=== smoke: compliance / deliveries ===")
    print("RULES:", list(sv.RULES))
    assert "normal" in sv.RULES and not sv.RULES["normal"].get("require_check")
    test_no_manifest_skipped()
    test_legacy_single_ok()
    test_multi_both_ok()
    test_multi_partial_fail_prefix()
    test_unknown_spec()
    test_cross_file_not_validated_as_export()
    test_runtime_hint()
    test_clear_manifest()
    test_finalize_matrix()
    test_phase2_force_manifest()
    test_cow_list_new_files()
    test_prompt_mentions_rules()
    print("=== ALL SMOKE PASSED ===")


if __name__ == "__main__":
    main()
