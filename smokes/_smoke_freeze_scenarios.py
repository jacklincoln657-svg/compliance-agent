"""架构冻结验收：评审要求的 6 个结构性场景。

运行（在项目根）: python smokes/_smoke_freeze_scenarios.py
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import json
import shutil
import tempfile
from pathlib import Path

import spec_validator as sv
from job_policy_state import append_policy_contexts_for_categories, clear_policy_contexts


def _export_pair(root: Path) -> list[str]:
    import csv

    d = root / "export_batch_v3"
    d.mkdir(parents=True, exist_ok=True)
    with (d / "clean_users_data.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Name", "Phone", "Score"])
        w.writerow(["Alice", "138****5678", "58.90"])
    (d / "audit_log.json").write_text(
        json.dumps({"total_rows": 1, "missing_scores_filled": 0}, ensure_ascii=False),
        encoding="utf-8",
    )
    return [
        "export_batch_v3/clean_users_data.csv",
        "export_batch_v3/audit_log.json",
    ]


def _thesis(root: Path) -> str:
    rel = "reports/开题报告.md"
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    body = (
        "一、毕业设计原始依据与背景：智慧医疗与信息孤岛、多模态数据。\n"
        "二、主要内容与研究内容：文本结构化、影像特征提取、跨模态检索。\n"
        "三、基本要求：提交可运行代码、答辩演示、隐私保护。\n"
        "四、进度安排：完成开题报告与任务书，后续中期检查。\n"
        + ("补充说明。" * 40)
    )
    path.write_text(body, encoding="utf-8")
    return rel


def scenario_1_no_rag() -> None:
    clear_policy_contexts()
    td = Path(tempfile.mkdtemp())
    try:
        (td / "chart.png").write_bytes(b"x")
        (td / sv.MANIFEST_NAME).write_text(
            json.dumps(
                {"deliveries": [{"files": ["chart.png"], "policy_refs": []}]},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        r = sv.finalize_job_delivery(str(td), new_files=["chart.png"])
        assert r.get("ok") and r.get("status") == "REGISTERED_NO_POLICY", r
        print("[OK] ① 无 RAG / refs=[] → REGISTERED_NO_POLICY")
    finally:
        shutil.rmtree(td, ignore_errors=True)


def scenario_2_rag_a() -> None:
    clear_policy_contexts()
    ctx = append_policy_contexts_for_categories(["engineering"])[0]
    td = Path(tempfile.mkdtemp())
    try:
        files = _export_pair(td)
        (td / sv.MANIFEST_NAME).write_text(
            json.dumps(
                {
                    "deliveries": [
                        {"files": files, "policy_refs": [ctx["ctx_id"]]},
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        r = sv.finalize_job_delivery(str(td), new_files=files)
        assert r.get("ok") and r.get("status") == "PASS", r
        print("[OK] ② RAG_A → refs=[ctx_A] → validator PASS")
    finally:
        shutil.rmtree(td, ignore_errors=True)
        clear_policy_contexts()


def scenario_3_two_files_two_ctx() -> None:
    clear_policy_contexts()
    c1 = append_policy_contexts_for_categories(["engineering"])[0]
    c2 = append_policy_contexts_for_categories(["academic"])[0]
    td = Path(tempfile.mkdtemp())
    try:
        files = _export_pair(td)
        md = _thesis(td)
        all_new = files + [md]
        (td / sv.MANIFEST_NAME).write_text(
            json.dumps(
                {
                    "deliveries": [
                        {"files": files, "policy_refs": [c1["ctx_id"]]},
                        {"files": [md], "policy_refs": [c2["ctx_id"]]},
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        r = sv.finalize_job_delivery(str(td), new_files=all_new)
        assert r.get("ok") and r.get("status") == "PASS", r
        assert len(r.get("results") or []) == 2
        print("[OK] ③ file_A→ctx_A, file_B→ctx_B")
    finally:
        shutil.rmtree(td, ignore_errors=True)
        clear_policy_contexts()


def scenario_4_multi_refs_one_delivery() -> None:
    clear_policy_contexts()
    c1 = append_policy_contexts_for_categories(["engineering"])[0]
    c2 = append_policy_contexts_for_categories(["academic"])[0]
    td = Path(tempfile.mkdtemp())
    try:
        files = _export_pair(td)
        md = _thesis(td)
        # 同一 delivery 绑两 ctx；files[0]=csv → thesis 弱闸失败 → 整单元 FAIL
        unit_files = files + [md]
        (td / sv.MANIFEST_NAME).write_text(
            json.dumps(
                {
                    "deliveries": [
                        {
                            "files": unit_files,
                            "policy_refs": [c1["ctx_id"], c2["ctx_id"]],
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        r = sv.finalize_job_delivery(str(td), new_files=unit_files)
        assert not r.get("ok"), r
        print("[OK] ④ multi refs：任一 validator FAIL → delivery FAIL")
    finally:
        shutil.rmtree(td, ignore_errors=True)
        clear_policy_contexts()


def scenario_5_fake_ctx() -> None:
    clear_policy_contexts()
    td = Path(tempfile.mkdtemp())
    try:
        (td / "a.txt").write_text("x", encoding="utf-8")
        (td / sv.MANIFEST_NAME).write_text(
            json.dumps(
                {
                    "deliveries": [
                        {"files": ["a.txt"], "policy_refs": ["ctx_ghost"]},
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        r = sv.finalize_job_delivery(str(td), new_files=["a.txt"])
        assert not r.get("ok") and r.get("status") == "FAIL", r
        print("[OK] ⑤ refs=[不存在的 ctx] → FAIL")
    finally:
        shutil.rmtree(td, ignore_errors=True)


def scenario_6_missing_manifest() -> None:
    clear_policy_contexts()
    td = Path(tempfile.mkdtemp())
    try:
        r = sv.finalize_job_delivery(str(td), new_files=["x.csv"])
        assert r.get("status") == "MISSING_MANIFEST" and not r.get("ok"), r
        print("[OK] ⑥ new_files 无 manifest → FAIL/MISSING_MANIFEST")
    finally:
        shutil.rmtree(td, ignore_errors=True)

    td = Path(tempfile.mkdtemp())
    try:
        (td / "a.txt").write_text("1", encoding="utf-8")
        (td / "b.txt").write_text("2", encoding="utf-8")
        (td / sv.MANIFEST_NAME).write_text(
            json.dumps(
                {"deliveries": [{"files": ["a.txt"], "policy_refs": []}]},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        r = sv.finalize_job_delivery(str(td), new_files=["a.txt", "b.txt"])
        assert not r.get("ok")
        assert any("未列入清单" in e for e in (r.get("errors") or [])), r
        print("[OK] ⑥ new_files 漏列 → FAIL")
    finally:
        shutil.rmtree(td, ignore_errors=True)


def main() -> None:
    print("=== freeze scenarios (architecture review) ===")
    scenario_1_no_rag()
    scenario_2_rag_a()
    scenario_3_two_files_two_ctx()
    scenario_4_multi_refs_one_delivery()
    scenario_5_fake_ctx()
    scenario_6_missing_manifest()
    print("=== ALL 6 FREEZE SCENARIOS PASSED ===")


if __name__ == "__main__":
    main()
