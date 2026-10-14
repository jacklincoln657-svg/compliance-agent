"""出仓等规范的硬校验：一台查核仪，按交付清单（manifest）选规则包。

不经 LLM。由 app._run_agent_job 后置钩子强制调用；模型跳过也无法出门。

触发条件：本轮任务写出 .compliance_manifest.json。
支持：
  - 单交付（兼容旧格式）: {"spec_id":"...","files":[...]}
  - 多交付: {"deliveries":[{"spec_id":"...","files":[...]}, ...]}
不再靠「猜固定 marker 文件名」或扫目录猜产物；也不根据 RAG 库名自动选规则。
"""
from __future__ import annotations

import csv
import json
import os
import re
from typing import Any, Callable

# Phone 掩码：138****5678
_MASKED_PHONE = re.compile(r"^\d{3}\*{4}\d{4}$")
# 疑似未脱敏 11 位手机号
_RAW_PHONE = re.compile(r"^1[3-9]\d{9}$")
# Score 原文必须两位小数（允许负号）
_SCORE_2DP = re.compile(r"^-?\d+\.\d{2}$")

# 相对 workspace 根；本轮任务开始时清空，有清单才验收
MANIFEST_NAME = ".compliance_manifest.json"


def manifest_path(workspace_dir: str) -> str:
    return os.path.join(workspace_dir, MANIFEST_NAME)


def clear_compliance_manifest(workspace_dir: str) -> None:
    """新任务开始时清掉旧清单，避免历史声明误触发。"""
    path = manifest_path(workspace_dir)
    try:
        if os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass


def _norm_rel(path: str) -> str:
    """清单内路径统一为正斜杠、相对路径。"""
    p = (path or "").strip().replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return p.lstrip("/")


def _parse_delivery_item(item: Any, *, index: int | None = None) -> dict[str, Any]:
    """
    解析单条交付单元（最小验收单元）。

    新格式（切片 B）：{"files": [...], "policy_refs": ["ctx_1"] | []}
      - policy_refs=[] → 无 Policy → normal（程序落点，非 Agent 填 normal）
    旧格式：{"spec_id": "...", "files": [...]}
    """
    prefix = f"deliveries[{index}] " if index is not None else ""
    if not isinstance(item, dict):
        return {
            "_invalid": True,
            "errors": [f"{MANIFEST_NAME} {prefix}必须是对象"],
        }

    files = item.get("files")
    if not isinstance(files, list) or not files:
        return {
            "_invalid": True,
            "errors": [f"{MANIFEST_NAME} {prefix}缺少非空数组 files"],
        }
    rels = [_norm_rel(str(x)) for x in files]
    if any(not r for r in rels):
        return {
            "_invalid": True,
            "errors": [f"{MANIFEST_NAME} {prefix}的 files 含空路径"],
        }

    has_refs_key = "policy_refs" in item
    if has_refs_key:
        refs_raw = item.get("policy_refs")
        if not isinstance(refs_raw, list):
            return {
                "_invalid": True,
                "errors": [f"{MANIFEST_NAME} {prefix}的 policy_refs 必须是数组"],
            }
        refs: list[str] = []
        for r in refs_raw:
            s = str(r).strip()
            if not s:
                return {
                    "_invalid": True,
                    "errors": [f"{MANIFEST_NAME} {prefix}的 policy_refs 含空值"],
                }
            refs.append(s)
        out: dict[str, Any] = {
            "files": rels,
            "policy_refs": refs,
            "uses_policy_refs": True,
        }
        # 若同时写了 spec_id，仅作提示字段保留，验收以 refs 为准
        if isinstance(item.get("spec_id"), str) and item["spec_id"].strip():
            out["spec_id_declared"] = item["spec_id"].strip()
        return out

    spec_id = item.get("spec_id")
    if not isinstance(spec_id, str) or not spec_id.strip():
        return {
            "_invalid": True,
            "errors": [
                f"{MANIFEST_NAME} {prefix}须提供 policy_refs（可空数组）或合法 spec_id"
            ],
        }
    return {
        "spec_id": spec_id.strip(),
        "files": rels,
        "uses_policy_refs": False,
    }


def load_compliance_manifest(workspace_dir: str) -> dict[str, Any] | None:
    """
    读取交付清单。
    成功返回 {"deliveries": [{"spec_id","files"}, ...]}；
    格式非法返回 {"_invalid": True, "errors": [...]}；
    无文件返回 None。
    """
    path = manifest_path(workspace_dir)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return {
            "_invalid": True,
            "errors": [f"{MANIFEST_NAME} 无法解析: {e}"],
        }
    if not isinstance(data, dict):
        return {
            "_invalid": True,
            "errors": [f"{MANIFEST_NAME} 必须是 JSON 对象"],
        }

    raw_deliveries = data.get("deliveries")
    if raw_deliveries is not None:
        if not isinstance(raw_deliveries, list) or not raw_deliveries:
            return {
                "_invalid": True,
                "errors": [f"{MANIFEST_NAME} 的 deliveries 必须是非空数组"],
            }
        deliveries: list[dict[str, Any]] = []
        for i, item in enumerate(raw_deliveries):
            parsed = _parse_delivery_item(item, index=i)
            if parsed.get("_invalid"):
                return parsed
            deliveries.append(parsed)
        return {"deliveries": deliveries}

    # 兼容旧单交付格式
    parsed = _parse_delivery_item(data, index=None)
    if parsed.get("_invalid"):
        return parsed
    return {"deliveries": [parsed]}


def _abs_under_workspace(workspace_dir: str, rel: str) -> str | None:
    """解析相对路径；禁止跳出 workspace。"""
    root = os.path.realpath(workspace_dir)
    abs_path = os.path.realpath(os.path.join(workspace_dir, rel.replace("/", os.sep)))
    if abs_path != root and not abs_path.startswith(root + os.sep):
        return None
    return abs_path


def _pick_by_basename(workspace_dir: str, files: list[str], basename: str) -> str | None:
    """从清单里按文件名取绝对路径（支持不同批次子目录）。"""
    for rel in files:
        if os.path.basename(rel.replace("\\", "/")) == basename:
            return _abs_under_workspace(workspace_dir, rel)
    return None


def _check_export_desensitize_v3(
    workspace_dir: str, files: list[str]
) -> list[str]:
    """对齐 knowledge_docs/engineering/pipeline_rules.txt（出仓脱敏 v3）底线条款。"""
    errors: list[str] = []
    csv_path = _pick_by_basename(workspace_dir, files, "clean_users_data.csv")
    audit_path = _pick_by_basename(workspace_dir, files, "audit_log.json")

    if csv_path is None:
        errors.append(
            "交付清单未声明 clean_users_data.csv（或路径跳出 workspace）"
        )
        return errors
    if audit_path is None:
        errors.append(
            "交付清单未声明 audit_log.json（或路径跳出 workspace）"
        )
        return errors
    if not os.path.isfile(csv_path):
        errors.append(f"清单声明的主表不存在: {csv_path}")
        return errors
    if not os.path.isfile(audit_path):
        errors.append(f"清单声明的审计文件不存在: {audit_path}")
        return errors

    # --- audit_log ---
    try:
        with open(audit_path, "r", encoding="utf-8") as f:
            audit = json.load(f)
    except Exception as e:
        errors.append(f"audit_log.json 无法解析: {e}")
        return errors

    if not isinstance(audit, dict):
        errors.append("audit_log.json 必须是 JSON 对象")
        return errors
    if "total_rows" not in audit:
        errors.append("audit_log 缺少必填字段 total_rows")
    if "missing_scores_filled" not in audit:
        errors.append("audit_log 缺少必填字段 missing_scores_filled")

    # --- CSV 原文（不用 pandas，避免 58.90→58.9）---
    try:
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.reader(f))
    except Exception as e:
        errors.append(f"无法读取 clean_users_data.csv: {e}")
        return errors

    if not rows:
        errors.append("clean_users_data.csv 为空")
        return errors

    header = rows[0]
    required_cols = ["Name", "Phone", "Score"]
    for col in required_cols:
        if col not in header:
            errors.append(f"CSV 缺少必填列（大小写敏感）: {col}")
    if any(c not in header for c in required_cols):
        return errors

    name_i = header.index("Name")
    phone_i = header.index("Phone")
    score_i = header.index("Score")
    data_rows = rows[1:]
    n = len(data_rows)

    if "total_rows" in audit:
        try:
            tr = int(audit["total_rows"])
            if tr != n:
                errors.append(f"行数不一致: CSV 数据行={n}, audit.total_rows={tr}")
        except (TypeError, ValueError):
            errors.append("audit.total_rows 必须是整数")

    phone_bad = 0
    score_bad = 0
    for idx, row in enumerate(data_rows, start=2):  # 文件行号（含表头为 1）
        if len(row) <= max(phone_i, score_i):
            errors.append(f"第 {idx} 行列数不足")
            continue
        phone = (row[phone_i] or "").strip()
        score = (row[score_i] or "").strip()
        if not phone:
            phone_bad += 1
            if phone_bad <= 3:
                errors.append(f"第 {idx} 行 Phone 为空")
        elif _RAW_PHONE.match(phone):
            phone_bad += 1
            if phone_bad <= 3:
                errors.append(f"第 {idx} 行 Phone 疑似未脱敏明文: {phone[:3]}********")
        elif not _MASKED_PHONE.match(phone):
            phone_bad += 1
            if phone_bad <= 3:
                errors.append(
                    f"第 {idx} 行 Phone 形态不符合 138****5678: {phone!r}"
                )
        if not score or score.lower() in {"nan", "none", "null"}:
            score_bad += 1
            if score_bad <= 3:
                errors.append(f"第 {idx} 行 Score 为空/缺失")
        elif not _SCORE_2DP.match(score):
            score_bad += 1
            if score_bad <= 3:
                errors.append(
                    f"第 {idx} 行 Score 原文不是两位小数: {score!r}（须如 58.90）"
                )

    if phone_bad > 3:
        errors.append(f"… Phone 问题共 {phone_bad} 行")
    if score_bad > 3:
        errors.append(f"… Score 问题共 {score_bad} 行")

    # 敏感列名不应残留
    for h in header:
        hl = h.lower()
        if "password" in hl or hl == "token" or "密钥" in h:
            errors.append(
                "CSV 仍含敏感列名（Password/Token/密钥），应删除并记入 dropped_sensitive_cols"
            )
            break

    return errors


def _check_thesis_outline_v1(workspace_dir: str, files: list[str]) -> list[str]:
    """
    任务书/开题类弱硬闸（对齐 academic 任务书结构底线，非全文语义审）。
    要求：声明的主文件存在、可读、有一定篇幅，并覆盖若干必备主题词。
    """
    errors: list[str] = []
    if not files:
        errors.append("thesis_outline_v1 交付清单 files 为空")
        return errors

    # 取清单第一个文件作为主交付（通常为开题报告.md）
    main_rel = files[0]
    main_path = _abs_under_workspace(workspace_dir, main_rel)
    if main_path is None:
        errors.append(f"清单路径非法（跳出 workspace）: {main_rel}")
        return errors
    if not os.path.isfile(main_path):
        errors.append(f"清单声明的文件不存在: {main_rel}")
        return errors

    try:
        with open(main_path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        errors.append(f"无法读取 {main_rel}: {e}")
        return errors

    body = (text or "").strip()
    if len(body) < 200:
        errors.append(f"{main_rel} 正文过短（<200 字），不像完整开题/任务书产物")

    # 弱约束：覆盖任务书常见板块关键词（满足多数即可）
    themes = [
        ("依据/背景", ("原始依据", "背景", "信息孤岛", "多模态")),
        ("研究内容", ("研究内容", "主要内容", "结构化", "特征提取")),
        ("基本要求", ("基本要求", "可运行", "答辩", "隐私")),
        ("进度安排", ("进度安排", "进度", "开题报告", "中期")),
    ]
    hit = 0
    missing_labels: list[str] = []
    lower = body  # 中文不 lower
    for label, keys in themes:
        if any(k in lower for k in keys):
            hit += 1
        else:
            missing_labels.append(label)
    if hit < 3:
        errors.append(
            f"{main_rel} 缺少任务书关键板块覆盖（已命中 {hit}/4，缺: "
            + "、".join(missing_labels)
            + "）"
        )
    return errors


# 规则注册表：合法 spec 的唯一中心（规范多了只加条目，不新建「查核仪」）
# require_check=False → 仅登记（REGISTERED），不跑字段硬校验
SpecCheck = Callable[[str, list[str]], list[str]]

RULES: dict[str, dict[str, Any]] = {
    "normal": {
        "check": None,
        "require_check": False,
        "label": "普通交付（仅清单登记，不硬验）",
        "required_basenames": [],
    },
    "export_desensitize_v3": {
        "check": _check_export_desensitize_v3,
        "require_check": True,
        "label": "CyberLobster 数据出仓脱敏规范_v3",
        # 清单里至少要能按 basename 找到这些文件
        "required_basenames": ["clean_users_data.csv", "audit_log.json"],
    },
    "thesis_outline_v1": {
        "check": _check_thesis_outline_v1,
        "require_check": True,
        "label": "毕业设计任务书/开题产物弱校验_v1",
        "required_basenames": [],
    },
}

def detect_pending_validation(workspace_dir: str) -> dict[str, Any] | None:
    """有合法交付清单则返回待验信息；无清单则不触发硬校验。"""
    raw = load_compliance_manifest(workspace_dir)
    if raw is None:
        return None
    if raw.get("_invalid"):
        return {
            "ok": False,
            "spec_id": None,
            "deliveries": [],
            "files": [],
            "label": MANIFEST_NAME,
            "errors": list(raw.get("errors") or []),
        }
    deliveries = list(raw.get("deliveries") or [])
    first = deliveries[0] if deliveries else {}
    return {
        "spec_id": first.get("spec_id"),
        "files": list(first.get("files") or []),
        "deliveries": deliveries,
        "label": None,
        "errors": [],
    }


# 兼容旧 import 名（已改为清单驱动）
def detect_active_spec_id(workspace_dir: str) -> str | None:
    pending = detect_pending_validation(workspace_dir)
    if not pending or pending.get("errors"):
        return None
    return pending.get("spec_id")


def _validate_one_delivery(
    workspace_dir: str, *, spec_id: str, files: list[str]
) -> dict[str, Any]:
    """对单条 delivery 跑 RULES[spec_id]。"""
    if spec_id not in RULES:
        return {
            "ok": False,
            "spec_id": spec_id,
            "files": files,
            "errors": [f"未知 spec_id: {spec_id}（未在 RULES 注册）"],
        }

    cfg = RULES[spec_id]
    path_errors: list[str] = []
    for rel in files:
        abs_p = _abs_under_workspace(workspace_dir, rel)
        if abs_p is None:
            path_errors.append(f"清单路径非法（跳出 workspace）: {rel}")
        elif not os.path.isfile(abs_p):
            path_errors.append(f"清单声明的文件不存在: {rel}")
    if path_errors:
        return {
            "ok": False,
            "spec_id": spec_id,
            "label": cfg.get("label") or spec_id,
            "files": files,
            "errors": path_errors,
        }

    for base in cfg.get("required_basenames") or []:
        if _pick_by_basename(workspace_dir, files, base) is None:
            path_errors.append(f"交付清单缺少规范要求的文件: {base}")
    if path_errors:
        return {
            "ok": False,
            "spec_id": spec_id,
            "label": cfg.get("label") or spec_id,
            "files": files,
            "errors": path_errors,
        }

    # 仅登记：合法 id + 文件存在即可（如 normal）
    if not cfg.get("require_check", True) or cfg.get("check") is None:
        return {
            "ok": True,
            "spec_id": spec_id,
            "label": cfg.get("label") or spec_id,
            "files": files,
            "errors": [],
            "status": "REGISTERED",
            "require_check": False,
        }

    check: SpecCheck = cfg["check"]
    errors = check(workspace_dir, files)
    return {
        "ok": len(errors) == 0,
        "spec_id": spec_id,
        "label": cfg.get("label") or spec_id,
        "files": files,
        "errors": errors,
        "status": "PASS" if not errors else "FAIL",
        "require_check": True,
    }


def resolve_policy_refs(policy_refs: list[str] | None) -> dict[str, Any]:
    """
    policy_refs → spec_ids。
    [] → normal（NO_POLICY）；非法 ctx_id → errors。
    """
    from job_policy_state import get_policy_context

    refs = list(policy_refs or [])
    if not refs:
        return {
            "ok": True,
            "spec_ids": ["normal"],
            "provenance": [],
            "errors": [],
        }

    errors: list[str] = []
    provenance: list[dict[str, Any]] = []
    spec_ids: list[str] = []
    seen: set[str] = set()
    for ref in refs:
        ctx = get_policy_context(ref)
        if ctx is None:
            errors.append(f"未知或本轮不存在的 policy_refs: `{ref}`")
            continue
        provenance.append(ctx)
        sid = str(ctx.get("spec_id") or "").strip()
        if not sid:
            errors.append(f"ctx `{ref}` 缺少 spec_id")
            continue
        if sid not in seen:
            seen.add(sid)
            spec_ids.append(sid)

    if errors:
        return {
            "ok": False,
            "spec_ids": spec_ids,
            "provenance": provenance,
            "errors": errors,
        }
    if not spec_ids:
        return {
            "ok": False,
            "spec_ids": [],
            "provenance": provenance,
            "errors": ["policy_refs 未能解析出任何 spec_id"],
        }
    return {
        "ok": True,
        "spec_ids": spec_ids,
        "provenance": provenance,
        "errors": [],
    }


def _validate_delivery_unit(
    workspace_dir: str,
    *,
    files: list[str],
    spec_ids: list[str],
    policy_refs: list[str] | None = None,
) -> dict[str, Any]:
    """
    一个最小验收单元：对每个 spec 依次校验，任一 FAIL → 单元 FAIL。
    """
    refs = list(policy_refs or [])
    sub_results: list[dict[str, Any]] = []
    all_errors: list[str] = []
    any_hard = False
    for sid in spec_ids:
        one = _validate_one_delivery(workspace_dir, spec_id=sid, files=files)
        sub_results.append(one)
        if one.get("require_check"):
            any_hard = True
        for e in one.get("errors") or []:
            all_errors.append(f"[{sid}] {e}")

    ok = all(bool(r.get("ok")) for r in sub_results)
    if ok:
        if not refs:
            # [] → NO_POLICY：声明无归属，不是「程序已证明无需硬验」
            status = "REGISTERED_NO_POLICY"
        elif any_hard:
            status = "PASS"
        else:
            status = "REGISTERED"
    else:
        status = "FAIL"

    label_parts = [
        str(r.get("label") or r.get("spec_id")) for r in sub_results if r.get("label")
    ]
    return {
        "ok": ok,
        "status": status,
        "spec_id": ",".join(spec_ids),
        "spec_ids": list(spec_ids),
        "policy_refs": refs,
        "no_policy": not refs,
        "label": "；".join(label_parts) if label_parts else ",".join(spec_ids),
        "files": files,
        "errors": all_errors,
        "sub_results": sub_results,
        "require_check": any_hard,
    }


def deliveries_use_policy_refs(deliveries: list[dict[str, Any]]) -> bool:
    """清单是否走 Policy Context 归属路径（任一条含 policy_refs 字段）。"""
    return any(bool(d.get("uses_policy_refs")) for d in deliveries)


def validate_compliance(
    workspace_dir: str,
    *,
    spec_id: str | None = None,
    files: list[str] | None = None,
) -> dict[str, Any]:
    """
    一台查核仪入口：优先用本轮交付清单（支持多 deliveries）；
    也可显式传入单个 spec_id/files。
    """
    if spec_id is not None and files is not None:
        one = _validate_one_delivery(workspace_dir, spec_id=spec_id, files=files)
        return {
            **one,
            "skipped": False,
            "results": [one],
        }

    pending = detect_pending_validation(workspace_dir)
    if pending is None:
        return {
            "ok": True,
            "spec_id": None,
            "skipped": True,
            "errors": [],
            "results": [],
        }
    if pending.get("errors"):
        return {
            "ok": False,
            "spec_id": pending.get("spec_id"),
            "label": pending.get("label"),
            "errors": pending["errors"],
            "skipped": False,
            "results": [],
        }

    deliveries = list(pending.get("deliveries") or [])
    if not deliveries:
        return {
            "ok": False,
            "spec_id": None,
            "errors": [f"{MANIFEST_NAME} 无有效 deliveries"],
            "skipped": False,
            "results": [],
        }

    if deliveries_use_policy_refs(deliveries):
        fin = _finalize_via_policy_refs(
            workspace_dir,
            deliveries,
            new_files=[],
        )
        return {
            "ok": fin.get("ok"),
            "spec_id": fin.get("spec_id"),
            "label": fin.get("label"),
            "errors": list(fin.get("errors") or []),
            "skipped": False,
            "results": list(fin.get("results") or []),
            "status": fin.get("status"),
            "provenance_mode": True,
        }

    results: list[dict[str, Any]] = []
    all_errors: list[str] = []
    labels: list[str] = []
    for d in deliveries:
        sid = str(d.get("spec_id") or "")
        fl = list(d.get("files") or [])
        one = _validate_one_delivery(workspace_dir, spec_id=sid, files=fl)
        results.append(one)
        lab = one.get("label") or sid
        if lab:
            labels.append(str(lab))
        for e in one.get("errors") or []:
            all_errors.append(f"[{sid}] {e}")

    ok = all(bool(r.get("ok")) for r in results)
    # 单交付时保持旧字段语义；多交付用合并 label
    if len(results) == 1:
        only = results[0]
        return {
            "ok": ok,
            "spec_id": only.get("spec_id"),
            "label": only.get("label"),
            "files": only.get("files"),
            "errors": list(only.get("errors") or []),
            "skipped": False,
            "results": results,
        }

    return {
        "ok": ok,
        "spec_id": ",".join(str(r.get("spec_id") or "") for r in results),
        "label": "；".join(labels) if labels else "多交付清单",
        "files": [f for r in results for f in (r.get("files") or [])],
        "errors": all_errors,
        "skipped": False,
        "results": results,
    }


def _classify_delivery_errors(result: dict[str, Any]) -> str:
    """粗分失败原因，便于提示用户/Agent。"""
    errs = " ".join(result.get("errors") or [])
    status = result.get("status") or ""
    if "未知或本轮不存在的 policy_refs" in errs or "未能解析出任何 spec_id" in errs:
        return "bad_policy_refs"
    if "未写入交付清单" in errs or "必须提交交付清单" in errs or status == "MISSING_MANIFEST":
        return "missing_manifest"
    if "未列入清单" in errs:
        return "files_not_listed"
    if "无法解析" in errs or "必须是 JSON" in errs or "缺少有效字段" in errs:
        return "manifest_format"
    if "未知 spec_id" in errs:
        return "unknown_spec"
    if "不存在" in errs or "缺少规范要求的文件" in errs:
        return "missing_files"
    if "Phone" in errs or "Score" in errs or "audit" in errs or "行数" in errs:
        return "field_rules"
    if "主题词" in errs or "篇幅" in errs or "thesis" in errs.lower():
        return "thesis_rules"
    if status == "FAIL":
        return "generic"
    return "generic"


def format_user_delivery_message(result: dict[str, Any]) -> str:
    """给聊天区看的人话摘要（短、可行动）。"""
    status = result.get("status") or ""
    label = result.get("label") or result.get("spec_id") or "交付"

    if status == "NO_FORMAL_DELIVERY" or result.get("skipped"):
        return ""  # 未声明正式交付，不必刷一条系统消息

    if status == "REGISTERED_NO_POLICY" and result.get("ok"):
        return (
            "✅ 本轮交付已按「无 Policy 归属」登记（`normal` / NO_POLICY）。\n"
            "含义：清单声明 policy_refs=[]，程序**未**证明「实际没用过规范库」；"
            "仅表示按声明走仅登记、不做字段硬验。"
            "若其实参考了检索到的规范，应把对应 ctx 写入 policy_refs。"
        )

    if status == "REGISTERED" and result.get("ok"):
        if result.get("legacy_compat"):
            return (
                f"✅ 本轮交付已登记（`{label}`，旧清单格式兼容）。\n"
                f"请尽快改为 deliveries + policy_refs；旧 spec_id 路径仅为迁移兼容。"
            )
        if result.get("provenance_mode"):
            return (
                f"✅ 本轮交付已登记（`{label}`）。\n"
                f"已绑定 Policy Context，该规则不要求字段硬校验。"
            )
        return (
            f"✅ 本轮交付已登记（`{label}`）。\n"
            f"当前为仅登记、不做字段硬校验。"
        )

    if status == "PASS" and result.get("ok"):
        if result.get("legacy_compat"):
            return (
                f"✅ 程序硬校验已通过（{label}，旧清单格式兼容）。\n"
                f"请尽快改为 policy_refs 归属；勿继续依赖自由填写的 spec_id。"
            )
        if result.get("provenance_mode"):
            return f"✅ 程序硬校验已通过（{label}）。交付单元与 Policy Context 归属一致。"
        return f"✅ 程序硬校验已通过（{label}）。"

    kind = _classify_delivery_errors(result)
    errs = result.get("errors") or []
    bullet = "\n".join(f"• {e}" for e in errs[:8])
    if len(errs) > 8:
        bullet += f"\n• …共 {len(errs)} 条"

    head = "⚠️ 交付验收未通过，请看下面原因和下一步。"
    tips = {
        "bad_policy_refs": (
            "原因类型：**清单里的 policy_refs 指向本轮不存在的 Policy Context**。\n"
            "下一步：只引用本轮检索成功后系统给出的 ctx_id；"
            "无知识库依赖的交付单元请写 `policy_refs: []`（程序落为 normal）。"
        ),
        "missing_manifest": (
            f"原因类型：**本轮已写出新文件，但未提交交付清单**（瞒报交付）。\n"
            f"下一步：必须在工作区根目录写入 `{MANIFEST_NAME}`，"
            "用 deliveries[]（files + policy_refs，可空=[]）；"
            "不要指望不写清单就能跳过验收。"
        ),
        "files_not_listed": (
            "原因类型：**本轮新文件未全部写入清单的 files**。\n"
            "下一步：把下列新文件补进清单，或删掉不应交付的临时产物后重试。"
        ),
        "manifest_format": (
            "原因类型：**交付清单格式不对**（JSON / 缺字段）。\n"
            f"下一步：在工作区根目录写好 `{MANIFEST_NAME}`，必须使用 "
            '`{"deliveries":[{"files":["相对路径"],"policy_refs":[]}]}`；'
            "有 Policy 时把检索回执里的 ctx_id 填进 policy_refs。"
            "仅写 spec_id 的旧格式仅为迁移兼容，新交付请勿再用。"
        ),
        "unknown_spec": (
            "原因类型：**清单里的 spec_id 未注册**（或旧格式兼容路径）。\n"
            "下一步：新清单请用 policy_refs；旧格式仅当过渡。"
        ),
        "missing_files": (
            "原因类型：**清单声明的文件缺失或不完整**。\n"
            "下一步：先生成清单里列出的文件，或改正 files 路径后再验收。"
        ),
        "field_rules": (
            "原因类型：**出仓字段未过硬规则**（如 Phone 掩码、Score 两位小数、audit 行数等）。\n"
            "下一步：按错误列表改 CSV/audit 后重写产物。"
        ),
        "thesis_rules": (
            "原因类型：**开题/任务书未过弱硬闸**（篇幅或必备主题等）。\n"
            "下一步：按错误补全文档内容后重新交付。"
        ),
        "generic": (
            "原因类型：交付验收失败。\n"
            "下一步：对照错误列表修改产物或清单（优先 policy_refs 归属）。"
        ),
    }
    tip = tips.get(kind, tips["generic"])
    return f"{head}\n\n{tip}\n\n具体错误：\n{bullet}"


def format_validation_feedback(result: dict[str, Any]) -> str:
    """塞回 Agent 的修复提示；要求用通俗中文向用户说明问题。"""
    kind = _classify_delivery_errors(result)
    user_view = format_user_delivery_message(result)

    lines = [
        "【交付验收未通过 — 必须修复产物/清单】",
        f"规范: {result.get('label') or result.get('spec_id')}",
        f"状态: {result.get('status') or 'FAIL'}",
        f"失败分类: {kind}",
    ]
    if result.get("provenance_mode") or kind in (
        "bad_policy_refs",
        "missing_manifest",
        "files_not_listed",
    ):
        lines.append(
            "优先用 deliveries[] + policy_refs（可空=[]→normal）；"
            "假 ctx_id 禁止编造。"
        )
    lines.append("错误列表:")
    for e in result.get("errors") or []:
        lines.append(f"- {e}")
    lines.extend(
        [
            "修复要求：",
            "1) 根据错误修改产物或清单；正式交付写入"
            f" `{MANIFEST_NAME}`，格式 "
            '{"deliveries":[{"files":[...],"policy_refs":["ctx_…"] 或 []}]}。',
            "2) 不要只改口头说明；policy_refs 必须 ⊆ 本轮真实 ctx。",
            "3) **回复用户时必须用通俗中文说明：问题出在哪一类"
            "（瞒报未写清单 / 假 policy_refs / 新文件未列入 files / 清单格式 / 缺文件 / 字段规则等）、"
            "以及你会改哪些文件或清单。**",
            "—— 以下可直接参考对用户说明 ——",
            user_view,
            "",
            format_policy_contexts_hint(),
        ]
    )
    return "\n".join(lines)


def format_policy_contexts_hint() -> str:
    """供 runtime / 修复轮注入：本轮已出现的候选 Policy Context。"""
    try:
        from job_policy_state import list_policy_contexts
    except ImportError:
        return "（本轮尚无 Policy Context）"
    ctxs = list_policy_contexts()
    if not ctxs:
        return (
            "本轮尚无 Policy Context（未成功检索带规则的知识库，"
            "或命中分类未在 KB_RULES 注册）。无 Policy 的交付单元请用 policy_refs: []。"
        )
    lines = ["本轮已出现的候选 Policy Context（attribution 只能引用下列 ctx_id）："]
    for c in ctxs:
        lines.append(
            f"- `{c.get('ctx_id')}` ← kb=`{c.get('kb')}` → spec=`{c.get('spec_id')}`"
            f" (rag={c.get('rag_call_id')})"
        )
    return "\n".join(lines)


def build_compliance_runtime_hint() -> str:
    """每轮 Job 注入：Policy Context 归属 + 清单写法。"""
    lines = [
        "【本轮交付验收 — Policy Context → 交付单元归属】",
        "search_knowledge_base 成功后，系统按命中知识库写入候选 ctx（见下文快照；检索回执也会带 ctx）。",
        f"有新文件必须写 `{MANIFEST_NAME}`，格式为 deliveries[]："
        '{"files":[...],"policy_refs":["ctx_…"] 或 []}。',
        "policy_refs=[] → 无 Policy → 程序落为 normal（仅登记，不硬验）。",
        "policy_refs 只能引用本轮真实 ctx；假 id → 验收 FAIL。",
        "一个 delivery = 最小验收单元（出仓 CSV+audit 必须同一条的 files）。",
        "多 policy_refs → 各 spec 依次硬验/登记，任一失败则该单元失败。",
        "硬不硬验由 RULES[spec].require_check 决定；不再使用页面「交付策略」预选。",
        "",
        format_policy_contexts_hint(),
        "",
        "RULES 摘要：",
    ]
    for sid, c in RULES.items():
        flag = "硬验" if c.get("require_check") else "仅登记/NO_POLICY"
        lines.append(f"- `{sid}`：{c.get('label') or sid}（{flag}）")
    return "\n".join(lines)


def finalize_job_delivery(
    workspace_dir: str,
    *,
    new_files: list[str] | None = None,
) -> dict[str, Any]:
    """
    Job 结束后置：

    - 有新文件且无清单 → MISSING_MANIFEST
    - 清单含 policy_refs → 对账 ctx → 按交付单元跑 RULES（[]→normal）
    - 仅旧字段 spec_id → 按声明的 spec 直接验收（迁移兼容）
    """
    new_files = [f.replace("\\", "/").lstrip("./") for f in (new_files or [])]
    pending = detect_pending_validation(workspace_dir)

    if pending is None:
        if new_files:
            preview = ", ".join(new_files[:12])
            more = f" 等共 {len(new_files)} 个" if len(new_files) > 12 else ""
            return {
                "ok": False,
                "status": "MISSING_MANIFEST",
                "skipped": False,
                "spec_id": None,
                "label": None,
                "new_files": new_files,
                "errors": [
                    f"本轮已写入新文件但未提交交付清单（必须写 `{MANIFEST_NAME}`）: {preview}{more}",
                    "清单须用 deliveries[]，每条含 files 与 policy_refs（可空=[]→normal）。",
                ],
                "results": [],
            }
        return {
            "ok": True,
            "status": "NO_FORMAL_DELIVERY",
            "skipped": True,
            "spec_id": None,
            "label": None,
            "errors": [],
            "results": [],
        }

    if pending.get("errors"):
        return {
            "ok": False,
            "status": "FAIL",
            "skipped": False,
            "spec_id": pending.get("spec_id"),
            "label": pending.get("label"),
            "errors": list(pending.get("errors") or []),
            "new_files": new_files,
            "results": [],
        }

    deliveries = list(pending.get("deliveries") or [])
    if not deliveries:
        return {
            "ok": False,
            "status": "FAIL",
            "skipped": False,
            "spec_id": None,
            "errors": [f"{MANIFEST_NAME} 无有效 deliveries"],
            "new_files": new_files,
            "results": [],
        }

    declared: set[str] = set()
    for d in deliveries:
        for f in d.get("files") or []:
            declared.add(str(f).replace("\\", "/").lstrip("./"))
    missing_in_manifest = [f for f in new_files if f not in declared]
    if missing_in_manifest:
        preview = ", ".join(missing_in_manifest[:12])
        more = f" 等共 {len(missing_in_manifest)} 个" if len(missing_in_manifest) > 12 else ""
        return {
            "ok": False,
            "status": "FAIL",
            "skipped": False,
            "spec_id": None,
            "errors": [
                f"本轮新文件未列入清单 files（须全部声明）: {preview}{more}",
            ],
            "new_files": new_files,
            "results": [],
        }

    if deliveries_use_policy_refs(deliveries):
        return _finalize_via_policy_refs(
            workspace_dir, deliveries, new_files=new_files
        )

    # 旧清单仅 spec_id：迁移兼容路径（非主路径；不代表可自由指定规则来源）
    LEGACY_SPEC_WARNING = (
        "【迁移兼容】清单仅含 spec_id、无 policy_refs："
        "此为旧格式兼容路径，不纳入新架构主链。"
        "请改为 deliveries[].files + policy_refs；"
        "规则来源应以本轮真实 Policy Context 为准，勿用自由填写的 spec_id 绕过 provenance。"
    )
    result = validate_compliance(workspace_dir)
    result["skipped"] = False
    result["new_files"] = new_files
    result["legacy_compat"] = True
    result["legacy_spec_id"] = True
    warns = list(result.get("warnings") or [])
    if LEGACY_SPEC_WARNING not in warns:
        warns.append(LEGACY_SPEC_WARNING)
    result["warnings"] = warns
    if result.get("ok"):
        statuses = [r.get("status") for r in (result.get("results") or [])]
        if statuses:
            if all(s == "REGISTERED" for s in statuses):
                result["status"] = "REGISTERED"
            elif all(s in ("PASS", "REGISTERED") for s in statuses):
                result["status"] = (
                    "PASS" if any(s == "PASS" for s in statuses) else "REGISTERED"
                )
            else:
                result["status"] = "FAIL" if not result.get("ok") else "PASS"
        else:
            any_hard = (RULES.get(str(result.get("spec_id") or "")) or {}).get(
                "require_check", True
            )
            result["status"] = "PASS" if any_hard else "REGISTERED"
    else:
        result["status"] = "FAIL"
    return result


def _finalize_via_policy_refs(
    workspace_dir: str,
    deliveries: list[dict[str, Any]],
    *,
    new_files: list[str],
) -> dict[str, Any]:
    """Policy Context 归属路径：refs 对账 → 按单元依次 validator。"""
    unit_results: list[dict[str, Any]] = []
    all_errors: list[str] = []

    for i, d in enumerate(deliveries):
        files = list(d.get("files") or [])
        if d.get("uses_policy_refs"):
            refs = list(d.get("policy_refs") or [])
            resolved = resolve_policy_refs(refs)
            if not resolved.get("ok"):
                for e in resolved.get("errors") or []:
                    all_errors.append(f"deliveries[{i}] {e}")
                unit_results.append(
                    {
                        "ok": False,
                        "status": "FAIL",
                        "files": files,
                        "policy_refs": refs,
                        "errors": list(resolved.get("errors") or []),
                    }
                )
                continue
            spec_ids = list(resolved["spec_ids"])
            unit = _validate_delivery_unit(
                workspace_dir,
                files=files,
                spec_ids=spec_ids,
                policy_refs=refs,
            )
            unit["provenance"] = resolved.get("provenance") or []
        else:
            sid = str(d.get("spec_id") or "").strip()
            unit = _validate_delivery_unit(
                workspace_dir,
                files=files,
                spec_ids=[sid] if sid else ["normal"],
                policy_refs=None,
            )

        unit_results.append(unit)
        if not unit.get("ok"):
            for e in unit.get("errors") or []:
                all_errors.append(f"deliveries[{i}] {e}")

    ok = all(bool(u.get("ok")) for u in unit_results)
    if any(not u.get("ok") for u in unit_results):
        ok = False

    if ok:
        # 全部单元均为无 Policy 声明 → REGISTERED_NO_POLICY（非「已证合规」）
        if unit_results and all(u.get("no_policy") for u in unit_results):
            status = "REGISTERED_NO_POLICY"
        elif any(u.get("require_check") for u in unit_results):
            status = "PASS"
        else:
            status = "REGISTERED"
    else:
        status = "FAIL"

    labels = [str(u.get("label") or u.get("spec_id") or "") for u in unit_results]
    spec_joined = ",".join(
        str(u.get("spec_id") or "") for u in unit_results if u.get("spec_id")
    )
    return {
        "ok": ok,
        "status": status,
        "skipped": False,
        "provenance_mode": True,
        "no_policy": bool(unit_results) and all(u.get("no_policy") for u in unit_results),
        "spec_id": spec_joined or None,
        "label": "；".join(x for x in labels if x) or None,
        "errors": all_errors,
        "new_files": new_files,
        "results": unit_results,
    }
