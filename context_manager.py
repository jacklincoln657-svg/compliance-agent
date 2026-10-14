"""上下文治理：模型摘要（方案 B）+ 会话归档。"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any

KEEP_RECENT_USER_TURNS = 4  # 保留窗口 Keep：摘要触发后留下的原文用户轮次数
MAX_RAW_USER_TURNS = 8  # 触发阈值 Max：相对水位线的原文用户轮次超过此值才摘要
# 兼容旧常量名（曾与 Keep 几乎重合，导致过线后每轮都摘要）
SUMMARIZE_AFTER_USER_TURNS = MAX_RAW_USER_TURNS

CLEAR_KEYWORDS = ("新任务", "清空上下文", "重新开始", "新会话")


def _role(msg: Any) -> str:
    if isinstance(msg, dict):
        return msg.get("role") or msg.get("type") or ""
    return getattr(msg, "type", "") or ""


def _content(msg: Any) -> str:
    if isinstance(msg, dict):
        return msg.get("content") or ""
    c = getattr(msg, "content", "") or ""
    return c if isinstance(c, str) else str(c)


def _tool_calls(msg: Any):
    if isinstance(msg, dict):
        return msg.get("tool_calls")
    return getattr(msg, "tool_calls", None) or None


def _is_user(msg: Any) -> bool:
    return _role(msg) in ("user", "human")


def _is_ai(msg: Any) -> bool:
    return _role(msg) in ("assistant", "ai")


def _is_tool(msg: Any) -> bool:
    return _role(msg) in ("tool",)


def to_wire_message(msg: Any) -> dict:
    if _is_user(msg):
        return {"role": "user", "content": _content(msg)}
    if _is_ai(msg):
        return {"role": "assistant", "content": _content(msg)}
    return {"role": "user", "content": _content(msg)}


def extract_visible_dialogue(history: list) -> list:
    """只保留用户话 + 最终助手回复（去掉 tool 轨迹）。"""
    visible = []
    for msg in history:
        if _is_tool(msg):
            continue
        if _is_ai(msg) and _tool_calls(msg):
            continue
        if _is_user(msg) or (_is_ai(msg) and _content(msg).strip()):
            visible.append(msg)
    return visible


def _format_visible_for_summary(msgs: list) -> str:
    lines = []
    for msg in msgs:
        who = "用户" if _is_user(msg) else "助手"
        text = _content(msg).strip()
        if len(text) > 1200:
            text = text[:1200] + "…(截断)"
        lines.append(f"{who}: {text}")
    return "\n".join(lines)


def summarize_with_llm(llm, older_text: str, prev_summary: str = "") -> str:
    """方案 B：再调一次模型，把旧对话压成短摘要。"""
    prompt = (
        "你是会话摘要器。请用中文输出不超过 10 条要点，覆盖："
        "用户目标、已做操作、生成的文件/路径、关键结论与未完成事项。"
        "不要废话，不要编号以外的开场白。\n\n"
    )
    if prev_summary.strip():
        prompt += f"【已有摘要】\n{prev_summary.strip()}\n\n"
    prompt += f"【需合并的更早对话】\n{older_text}\n"

    try:
        resp = llm.invoke(prompt)
        text = getattr(resp, "content", None) or str(resp)
        if isinstance(text, list):
            text = "".join(str(x) for x in text)
        return (text or "").strip() or prev_summary
    except Exception as e:
        fallback = prev_summary.strip()
        note = f"(摘要失败: {e})"
        return f"{fallback}\n{note}".strip() if fallback else note


def build_model_context(
    full_history: list,
    summary: str,
    llm,
    keep_recent: int = KEEP_RECENT_USER_TURNS,
    max_raw: int = MAX_RAW_USER_TURNS,
    summary_covers: int = 0,
) -> tuple[list, str, int]:
    """
    弹性滑动窗口：构造发给 Agent 的瘦身 messages，并可能更新 summary。

    - 静默期：相对水位线的原文用户轮次 ≤ Max → 不调摘要 LLM
    - 触发期：原文用户轮次 > Max → 只把「未覆盖且不在 Keep 内」的那段并入摘要，
      水位线推进到 n_user - Keep；之后再静默增长，避免每轮重压

    返回 (model_messages, new_summary, new_summary_covers)。
    summary_covers = 摘要已覆盖的「可见对话中的用户轮次数」（从最早一轮起算）。
    """
    visible = extract_visible_dialogue(full_history)
    if not visible:
        return [], summary or "", max(0, int(summary_covers or 0))

    user_positions = [i for i, m in enumerate(visible) if _is_user(m)]
    n_user = len(user_positions)
    new_summary = summary or ""
    covered = max(0, int(summary_covers or 0))

    # 旧会话无水位线：假定摘要已覆盖「除最近 Keep 外」的轮次，避免升级后每轮全量重压
    if covered <= 0 and new_summary.strip() and n_user > keep_recent:
        covered = n_user - keep_recent
    covered = min(covered, n_user)

    raw_count = n_user - covered

    if raw_count > max_raw and n_user > keep_recent:
        split_at = user_positions[-keep_recent]
        older_start = user_positions[covered] if covered > 0 else 0
        older = visible[older_start:split_at]
        recent = visible[split_at:]
        if older:
            older_text = _format_visible_for_summary(older)
            new_summary = summarize_with_llm(llm, older_text, new_summary)
        covered = n_user - keep_recent
    elif covered > 0 and covered < n_user:
        recent = visible[user_positions[covered] :]
    else:
        recent = visible

    model_msgs: list = []
    if new_summary.strip():
        model_msgs.append({
            "role": "user",
            "content": (
                "[会话摘要 — 更早对话的压缩记忆，请结合后续消息继续任务]\n"
                + new_summary.strip()
            ),
        })
        model_msgs.append({
            "role": "assistant",
            "content": "好的，我已记住摘要中的背景，请继续。",
        })

    model_msgs.extend(to_wire_message(m) for m in recent)
    return model_msgs, new_summary, covered


def wants_new_session(text: str) -> bool:
    return any(k in (text or "") for k in CLEAR_KEYWORDS)


def strip_clear_keywords(text: str) -> str:
    rest = text or ""
    for k in CLEAR_KEYWORDS:
        rest = rest.replace(k, "")
    return rest.strip(" \t\n\r：:，,.-")


def _sanitize_topic(topic: str) -> str:
    """做成可放进文件名的短主题。"""
    topic = (topic or "").strip()
    topic = re.sub(r'[\\/:*?"<>|\s]+', "_", topic)
    topic = re.sub(r"_+", "_", topic).strip("._")
    if len(topic) > 24:
        topic = topic[:24].rstrip("._")
    return topic or "未命名"


def derive_session_topic(history: list, summary: str, llm=None) -> str:
    """从摘要或首条用户话提炼主题；有 llm 时再压成短中文名。"""
    seed = ""
    if (summary or "").strip():
        seed = summary.strip().splitlines()[0].strip()
        seed = re.sub(r"^[\d\-、.．)）]+\s*", "", seed)
    if not seed:
        for msg in history:
            if _is_user(msg):
                seed = _content(msg).strip().splitlines()[0]
                break
    if not seed:
        return "未命名"

    if llm is not None:
        try:
            resp = llm.invoke(
                "把下面内容概括成4到12个字的中文主题名，只输出主题本身，"
                "不要引号、标点或解释：\n" + seed[:500]
            )
            text = getattr(resp, "content", None) or str(resp)
            if isinstance(text, list):
                text = "".join(str(x) for x in text)
            text = (text or "").strip().splitlines()[0].strip()
            if text:
                return _sanitize_topic(text)
        except Exception:
            pass

    return _sanitize_topic(seed[:12])


def archive_session(
    history: list,
    summary: str,
    workspace_dir: str,
    llm=None,
    summary_covers: int = 0,
) -> str | None:
    """把当前会话落到 workspace/_sessions/session_时间_主题.json。"""
    if not history and not (summary or "").strip():
        return None

    sessions_dir = os.path.join(workspace_dir, "_sessions")
    os.makedirs(sessions_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    topic = derive_session_topic(history, summary, llm=llm)
    filename = f"session_{ts}_{topic}.json"
    path = os.path.join(sessions_dir, filename)

    serializable = []
    for msg in history:
        item = {"role": _role(msg), "content": _content(msg)}
        tc = _tool_calls(msg)
        if tc:
            item["tool_calls"] = str(tc)[:2000]
        serializable.append(item)

    payload = {
        "archived_at": ts,
        "topic": topic,
        "summary": summary or "",
        "summary_covers_user_turns": max(0, int(summary_covers or 0)),
        "message_count": len(history),
        "messages": serializable,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return f"_sessions/{filename}"


ACTIVE_SESSION_FILE = "_active.json"


def _active_path(workspace_dir: str) -> str:
    sessions_dir = os.path.join(workspace_dir, "_sessions")
    os.makedirs(sessions_dir, exist_ok=True)
    return os.path.join(sessions_dir, ACTIVE_SESSION_FILE)


def _serialize_history(history: list) -> list:
    serializable = []
    for msg in history:
        item = {"role": _role(msg), "content": _content(msg)}
        tc = _tool_calls(msg)
        if tc:
            # 保留结构便于恢复；过长则截断字符串
            try:
                item["tool_calls"] = tc
            except Exception:
                item["tool_calls"] = str(tc)[:2000]
        serializable.append(item)
    return serializable


def save_active_session(
    history: list,
    summary: str,
    workspace_dir: str,
    summary_covers: int = 0,
) -> None:
    """进行中草稿落盘（防进程崩溃）。浏览器刷新由 UI 自动归档到历史后再开空白会话。"""
    path = _active_path(workspace_dir)
    payload = {
        "summary": summary or "",
        "summary_covers_user_turns": max(0, int(summary_covers or 0)),
        "message_count": len(history),
        "messages": _serialize_history(history),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_active_session(workspace_dir: str) -> tuple[list, str, int]:
    """读取进行中草稿（刷新归档前 / 新会话前）→ (history, summary, summary_covers)。"""
    path = _active_path(workspace_dir)
    if not os.path.isfile(path):
        return [], "", 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        messages = data.get("messages") or []
        summary = data.get("summary") or ""
        covers = int(data.get("summary_covers_user_turns") or 0)
        # 统一成 dict，后续逻辑已兼容
        history = []
        for m in messages:
            if not isinstance(m, dict):
                continue
            history.append(m)
        return history, summary, covers
    except Exception:
        return [], "", 0


def clear_active_session(workspace_dir: str) -> None:
    path = _active_path(workspace_dir)
    if os.path.isfile(path):
        try:
            os.remove(path)
        except OSError:
            pass


def delete_archived_session(workspace_dir: str, filename: str) -> bool:
    """删除一条归档会话文件。成功返回 True。"""
    safe = os.path.basename(filename or "")
    if not safe.startswith("session_") or not safe.endswith(".json"):
        return False
    path = os.path.join(workspace_dir, "_sessions", safe)
    if not os.path.isfile(path):
        return False
    try:
        os.remove(path)
        return True
    except OSError:
        return False


def list_archived_sessions(workspace_dir: str, limit: int = 40) -> list[dict]:
    """列出已归档会话（供左侧历史栏）。不含 _active.json。"""
    sessions_dir = os.path.join(workspace_dir, "_sessions")
    if not os.path.isdir(sessions_dir):
        return []
    items = []
    for name in os.listdir(sessions_dir):
        if not name.startswith("session_") or not name.endswith(".json"):
            continue
        path = os.path.join(sessions_dir, name)
        topic = "未命名"
        archived_at = ""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            topic = data.get("topic") or topic
            archived_at = data.get("archived_at") or ""
        except Exception:
            parts = name[len("session_") : -len(".json")].split("_", 2)
            if len(parts) >= 3:
                archived_at = f"{parts[0]}_{parts[1]}"
                topic = parts[2]
            elif len(parts) == 2:
                archived_at = f"{parts[0]}_{parts[1]}"
        items.append(
            {
                "filename": name,
                "topic": topic,
                "archived_at": archived_at,
                "mtime": os.path.getmtime(path),
            }
        )
    items.sort(key=lambda x: x["mtime"], reverse=True)
    return items[:limit]


def load_archived_session(
    workspace_dir: str, filename: str
) -> tuple[list, str, str, int]:
    """加载归档会话 → (history, summary, topic, summary_covers)。"""
    safe = os.path.basename(filename or "")
    if not safe.startswith("session_") or not safe.endswith(".json"):
        return [], "", "", 0
    path = os.path.join(workspace_dir, "_sessions", safe)
    if not os.path.isfile(path):
        return [], "", "", 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        history = [m for m in (data.get("messages") or []) if isinstance(m, dict)]
        summary = data.get("summary") or ""
        topic = data.get("topic") or "未命名"
        covers = int(data.get("summary_covers_user_turns") or 0)
        return history, summary, topic, covers
    except Exception:
        return [], "", "", 0
