import os
import base64
import threading
from urllib.parse import quote

import dash
import pandas as pd
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, ALL, ctx, dash_table, no_update
from flask import abort, send_from_directory, Response
from langchain_core.callbacks import BaseCallbackHandler

from agent_core import get_agent, get_llm, llm, model_label
from tools_sandbox import (
    WORKSPACE_DIR,
    begin_cow_session,
    build_kb_runtime_hint,
    clear_kb_scope,
    end_cow_session,
    get_cow_session,
    list_knowledge_categories,
    request_kill_sandbox,
    set_cancel_checker,
    set_kb_scope,
)
from context_manager import (
    archive_session,
    build_model_context,
    clear_active_session,
    delete_archived_session,
    list_archived_sessions,
    load_active_session,
    load_archived_session,
    save_active_session,
    strip_clear_keywords,
    wants_new_session,
)
from spec_validator import (
    clear_compliance_manifest,
    format_validation_feedback,
    format_user_delivery_message,
    format_policy_contexts_hint,
    finalize_job_delivery,
    build_compliance_runtime_hint,
)

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=Sora:wght@500;600;700&display=swap",
    ],
)

app.index_string = """<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>CyberLobster · 数据分析</title>
{%favicon%}
{%css%}
<style>
  :root {
    --ink: #16232e;
    --ink-soft: #5a6b78;
    --line: #d5dee6;
    --panel: #ffffff;
    --wash: #eef3f6;
    --accent: #0c6e73;
    --accent-deep: #085458;
    --accent-soft: #d7ecee;
    --user-bg: #e8f4f4;
    --danger: #b33a3a;
    --radius: 12px;
    --font: "IBM Plex Sans", "Segoe UI", sans-serif;
    --display: "Sora", "IBM Plex Sans", sans-serif;
  }
  html, body {
    height: 100%;
    margin: 0;
    font-family: var(--font);
    color: var(--ink);
    background:
      radial-gradient(ellipse 80% 50% at 10% -10%, #d9ecec 0%, transparent 55%),
      radial-gradient(ellipse 70% 45% at 100% 0%, #e4ecf4 0%, transparent 50%),
      linear-gradient(180deg, #f3f6f8 0%, #e9eef2 100%);
    background-attachment: fixed;
  }
  .app-shell {
    max-width: 1480px;
    margin: 0 auto;
    padding: 18px 20px 28px;
  }
  .brand-bar {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 16px;
    padding: 4px 2px 12px;
    border-bottom: 1px solid rgba(12, 110, 115, 0.18);
  }
  .brand-name {
    font-family: var(--display);
    font-weight: 700;
    font-size: 1.55rem;
    letter-spacing: -0.03em;
    color: var(--ink);
    margin: 0;
  }
  .brand-name span {
    color: var(--accent);
  }
  .brand-tag {
    font-size: 0.84rem;
    color: var(--ink-soft);
    font-weight: 500;
  }
  .panel {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    height: calc(100vh - 110px);
    min-height: 560px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  .panel-head {
    padding: 12px 14px;
    border-bottom: 1px solid var(--line);
    background: linear-gradient(180deg, #fbfcfd, #f5f8fa);
    font-family: var(--display);
    font-weight: 600;
    font-size: 0.92rem;
    letter-spacing: -0.01em;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
  }
  .panel-body {
    flex: 1;
    min-height: 0;
    overflow: auto;
    padding: 12px;
  }
  .panel-foot {
    border-top: 1px solid var(--line);
    padding: 12px;
    background: #f7fafb;
  }
  .chat-scroll {
    background:
      linear-gradient(180deg, rgba(255,255,255,0.4), transparent 40px),
      var(--wash);
    border-radius: 8px;
    padding: 14px 12px;
    height: 100%;
  }
  .msg-row { margin-bottom: 14px; }
  .msg-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--ink-soft);
    margin-bottom: 4px;
  }
  .user-bubble {
    white-space: pre-wrap;
    word-break: break-word;
    line-height: 1.55;
    background: var(--user-bg);
    border: 1px solid #c5dede;
    border-radius: 12px 12px 4px 12px;
    padding: 10px 12px;
    color: var(--ink);
    font-weight: 500;
  }
  .ai-bubble {
    white-space: pre-wrap;
    word-break: break-word;
    line-height: 1.6;
    background: #fff;
    border: 1px solid var(--line);
    border-radius: 12px 12px 12px 4px;
    padding: 10px 12px;
    color: var(--ink);
  }
  .ai-bubble p { margin: 0 0 0.75em; }
  .ai-bubble p:last-child { margin-bottom: 0; }
  .status-line {
    font-size: 0.8rem;
    color: var(--ink-soft);
    margin-top: 8px;
    min-height: 1.2em;
  }
  .composer-field label {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--ink-soft);
    margin-bottom: 4px;
  }
  .composer-field .form-select {
    border-color: var(--line);
    border-radius: 8px;
    font-size: 0.85rem;
    background-color: #fff;
  }
  .upload-zone {
    border: 1px dashed #b7c8d0;
    border-radius: 8px;
    padding: 10px;
    text-align: center;
    cursor: pointer;
    background: #fff;
    color: var(--ink-soft);
    font-size: 0.82rem;
    margin-bottom: 10px;
    transition: border-color .15s, background .15s;
  }
  .upload-zone:hover {
    border-color: var(--accent);
    background: var(--accent-soft);
  }
  .thinking-btn {
    border-radius: 8px !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    border-color: var(--line) !important;
    color: var(--ink-soft) !important;
    background: #fff !important;
  }
  .thinking-btn.on {
    background: var(--accent-soft) !important;
    border-color: var(--accent) !important;
    color: var(--accent-deep) !important;
  }
  .btn-primary {
    background: var(--accent) !important;
    border-color: var(--accent-deep) !important;
    font-weight: 600 !important;
  }
  .btn-primary:hover, .btn-primary:focus {
    background: var(--accent-deep) !important;
    border-color: var(--accent-deep) !important;
  }
  .hist-section {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: var(--ink-soft);
    margin: 10px 0 6px;
  }
  .hist-row {
    display: flex;
    align-items: stretch;
    gap: 4px;
    margin-bottom: 4px;
  }
  .hist-open-btn {
    flex: 1;
    text-align: left !important;
    border: 1px solid transparent !important;
    background: transparent !important;
    border-radius: 8px !important;
    padding: 8px 10px !important;
    color: var(--ink) !important;
  }
  .hist-open-btn:hover {
    background: var(--wash) !important;
  }
  .hist-open-btn.active-item {
    background: var(--accent-soft) !important;
    border-color: #b5d6d8 !important;
  }
  .hist-title {
    font-weight: 600;
    font-size: 0.86rem;
    line-height: 1.25;
  }
  .hist-del-btn {
    width: 28px;
    border: none;
    border-radius: 6px;
    background: transparent;
    color: #9aa8b3;
    cursor: pointer;
    opacity: 0.35;
  }
  .hist-row:hover .hist-del-btn { opacity: 1; }
  .hist-del-btn:hover { background: #fde8e8; color: var(--danger); }
  .fs-btn {
    width: 100%;
    text-align: left !important;
    border: 1px solid var(--line) !important;
    background: #fff !important;
    border-radius: 8px !important;
    margin-bottom: 6px !important;
    padding: 8px 10px !important;
    color: var(--ink) !important;
    font-size: 0.86rem !important;
  }
  .fs-btn:hover { border-color: var(--accent) !important; background: var(--accent-soft) !important; }
  .fs-btn.fs-file {
    border-style: dashed !important;
    background: transparent !important;
  }
  .fs-crumb {
    font-size: 0.8rem;
    color: var(--ink-soft);
    margin-bottom: 8px;
    font-family: ui-monospace, "Cascadia Code", monospace;
  }
  .fs-crumb strong { color: var(--ink); font-weight: 600; }
  .preview-empty {
    color: var(--ink-soft);
    font-size: 0.9rem;
    padding: 24px 8px;
    text-align: center;
  }
  .input-group .form-control {
    border-color: var(--line);
    border-radius: 8px 0 0 8px !important;
  }
  .input-group .btn { font-weight: 600; }
  @media (max-width: 992px) {
    .panel { height: auto; min-height: 420px; margin-bottom: 12px; }
  }
</style>
</head>
<body>
{%app_entry%}
<footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>
"""

# 主界面默认空白（与 fusion 一致）：刷新 = 新会话；草稿先归档再清空。
chat_history: list = []
conversation_summary = ""
conversation_summary_covers = 0  # 摘要已覆盖的用户轮次数（弹性窗口水位线）

_job_lock = threading.Lock()
_job = {
    "running": False,
    "error": None,
    "cancel": threading.Event(),
    "thinking": False,
}
# 新会话 / 刷新归档后递增，作废进行中的后台任务写盘；loaded_from=正在浏览的历史文件
_nav = {"loaded_from": None, "session_epoch": 0}
_session_epoch = 0  # 与 _nav["session_epoch"] 同步，兼容旧引用


def _bump_epoch():
    global _session_epoch
    _nav["session_epoch"] = int(_nav.get("session_epoch") or 0) + 1
    _session_epoch = _nav["session_epoch"]
    _nav["loaded_from"] = None


def _archive_and_reset_to_blank(*, cancel_job: bool = True) -> str | None:
    """归档当前草稿并清空对话区。返回归档相对路径（若有）。"""
    global chat_history, conversation_summary, conversation_summary_covers
    with _job_lock:
        if cancel_job and _job["running"]:
            cancel = _job.get("cancel")
            if cancel:
                cancel.set()
            request_kill_sandbox()
        rel = None
        if chat_history and not _nav.get("loaded_from"):
            rel = archive_session(
                chat_history,
                conversation_summary,
                WORKSPACE_DIR,
                llm=llm,
                summary_covers=conversation_summary_covers,
            )
        elif not _nav.get("loaded_from"):
            draft_h, draft_s, draft_c = load_active_session(WORKSPACE_DIR)
            if draft_h:
                rel = archive_session(
                    draft_h, draft_s, WORKSPACE_DIR, llm=llm, summary_covers=draft_c
                )
        chat_history = []
        conversation_summary = ""
        conversation_summary_covers = 0
        _bump_epoch()
        clear_active_session(WORKSPACE_DIR)
        save_active_session([], "", WORKSPACE_DIR, summary_covers=0)
        _job["running"] = False
    return rel


_SKIP_NAMES = {"temp_sandbox_script.py"}
_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}
_TABLE_EXT = {".csv", ".xlsx", ".xls"}
_TEXT_EXT = {".txt", ".md", ".json", ".py", ".log", ".tsv"}

INBOX_DIR = os.path.join(WORKSPACE_DIR, "_inbox")
os.makedirs(INBOX_DIR, exist_ok=True)

# 面板预览行数上限（过大易卡）；新窗口可看全表交互
_TABLE_PREVIEW_MAX_ROWS = 2000


def _prepare_table_df(df: pd.DataFrame) -> pd.DataFrame:
    """NaN → None，列名转 str，方便 DataTable / JSON。"""
    out = df.copy()
    out.columns = [str(c) for c in out.columns]
    return out.astype(object).where(pd.notnull(out), None)


def build_dash_datatable(df: pd.DataFrame, page_size: int = 12) -> dash_table.DataTable:
    """右侧预览：原生排序 + 列筛选 + 分页。"""
    prep = _prepare_table_df(df)
    records = [{str(k): v for k, v in row.items()} for row in prep.to_dict("records")]
    return dash_table.DataTable(
        data=records,
        columns=[{"name": c, "id": c} for c in prep.columns],
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        page_action="native",
        page_size=page_size,
        filter_options={"case": "insensitive"},
        style_table={"overflowX": "auto", "maxHeight": "52vh", "overflowY": "auto"},
        style_cell={
            "textAlign": "center",
            "padding": "6px",
            "minWidth": "72px",
            "maxWidth": "220px",
            "whiteSpace": "normal",
        },
        style_header={"backgroundColor": "#e9ecef", "fontWeight": "bold"},
        style_filter={"backgroundColor": "#ffffff"},
    )


def build_standalone_table_page(title: str, df: pd.DataFrame) -> str:
    """
    新窗口独立页：Bootstrap + DataTables 标准交互表
    （全局搜索 / 列筛选 / 排序 / 分页），不依赖 Dash。
    """
    from html import escape

    prep = _prepare_table_df(df)
    columns = [str(c) for c in prep.columns]

    thead_cols = "".join(f"<th>{escape(c)}</th>" for c in columns)
    body_rows = []
    for rec in prep.to_dict("records"):
        cells = []
        for c in columns:
            v = rec.get(c)
            cells.append(f"<td>{escape('' if v is None else str(v))}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    tbody = "\n".join(body_rows)
    safe_title = escape(title)
    n_rows = len(prep)

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{safe_title}</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"/>
<link rel="stylesheet" href="https://cdn.datatables.net/1.13.8/css/dataTables.bootstrap5.min.css"/>
<style>
  body {{ background: #f8f9fa; }}
  .page {{ max-width: 1280px; margin: 0 auto; padding: 24px 16px 40px; }}
  .page h1 {{ font-size: 1.25rem; margin-bottom: 4px; }}
  .page .hint {{ color: #6c757d; font-size: 0.875rem; margin-bottom: 16px; }}
  .card-table {{ background: #fff; border: 1px solid #dee2e6; border-radius: .5rem; padding: 16px; }}
  table.dataTable thead th {{ white-space: nowrap; vertical-align: middle; }}
  table.dataTable thead tr.filters th {{ padding: 6px 8px; background: #fff; font-weight: 400; }}
  table.dataTable thead tr.filters input {{
    width: 100%; min-width: 72px; box-sizing: border-box;
    padding: .25rem .5rem; font-size: .875rem; font-weight: 400;
    border: 1px solid #ced4da; border-radius: .375rem; background: #fff;
  }}
  table.dataTable thead tr.filters input:focus {{
    border-color: #86b7fe; outline: 0;
    box-shadow: 0 0 0 .2rem rgba(13,110,253,.25);
  }}
  table.dataTable tbody td {{ vertical-align: middle; }}
</style>
</head>
<body>
<div class="page">
  <h1 class="fw-semibold">{safe_title}</h1>
  <p class="hint">共 {n_rows} 行 · 表头排序 · 列下输入框筛选 · 右上角全局搜索</p>
  <div class="card-table table-responsive">
    <table id="data-table" class="table table-striped table-hover table-bordered table-sm w-100">
      <thead>
        <tr>{thead_cols}</tr>
      </thead>
      <tbody>
        {tbody}
      </tbody>
    </table>
  </div>
</div>
<script src="https://cdn.jsdelivr.net/npm/jquery@3.7.1/dist/jquery.min.js"></script>
<script src="https://cdn.datatables.net/1.13.8/js/jquery.dataTables.min.js"></script>
<script src="https://cdn.datatables.net/1.13.8/js/dataTables.bootstrap5.min.js"></script>
<script>
$(function () {{
  const $table = $("#data-table");
  // DataTables 官方列筛选写法：复制表头一行，不带事件，避免点输入框触发排序
  $table.find("thead tr").clone().addClass("filters").appendTo($table.find("thead"));
  $table.find("thead tr.filters th").each(function () {{
    const title = $(this).text().trim();
    $(this)
      .removeClass("sorting sorting_asc sorting_desc")
      .html(
        '<input type="search" class="form-control form-control-sm" placeholder="' +
          title +
          '" aria-label="筛选 ' +
          title +
          '" />'
      );
  }});

  const table = $table.DataTable({{
    orderCellsTop: true,
    pageLength: 25,
    lengthMenu: [[10, 25, 50, 100, -1], [10, 25, 50, 100, "全部"]],
    order: [],
    autoWidth: false,
    language: {{
      search: "搜索:",
      lengthMenu: "每页显示 _MENU_ 条",
      info: "显示第 _START_ 至 _END_ 条，共 _TOTAL_ 条",
      infoEmpty: "暂无数据",
      infoFiltered: "（从 _MAX_ 条中筛选）",
      zeroRecords: "没有匹配的记录",
      paginate: {{
        first: "首页",
        last: "末页",
        next: "下一页",
        previous: "上一页"
      }}
    }}
  }});

  table.columns().every(function (i) {{
    const col = this;
    $("input", $table.find("thead tr.filters th").eq(i)).on("keyup change clear", function (e) {{
      e.stopPropagation();
      if (col.search() !== this.value) {{
        col.search(this.value).draw();
      }}
    }});
  }});
}});
</script>
</body>
</html>
"""


def rel_workspace(path: str) -> str:
    return os.path.relpath(path, WORKSPACE_DIR).replace("\\", "/")


def resolve_workspace_path(rel_path: str) -> str:
    rel = (rel_path or "").replace("\\", "/").lstrip("/")
    full = os.path.normpath(os.path.join(WORKSPACE_DIR, *rel.split("/"))) if rel else os.path.normpath(WORKSPACE_DIR)
    root = os.path.normpath(WORKSPACE_DIR)
    if full != root and not full.startswith(root + os.sep):
        abort(404)
    return full


def join_rel(cwd: str, name: str) -> str:
    cwd = (cwd or "").replace("\\", "/").strip("/")
    return f"{cwd}/{name}".strip("/") if cwd else name


def list_dir(cwd_rel: str):
    """列出当前相对目录下的文件夹与文件。"""
    abs_dir = resolve_workspace_path(cwd_rel or "")
    if not os.path.isdir(abs_dir):
        return [], []
    dirs, files = [], []
    for name in os.listdir(abs_dir):
        if name.startswith(".") or name in _SKIP_NAMES:
            continue
        if name == "_active.json":
            continue  # 进行中会话缓存，不在浏览器里展示
        path = os.path.join(abs_dir, name)
        if os.path.isdir(path):
            dirs.append(name)
        elif os.path.isfile(path):
            files.append(name)
    dirs.sort(key=str.lower)
    files.sort(key=str.lower)
    return dirs, files


def build_browser(cwd_rel: str):
    cwd_rel = (cwd_rel or "").replace("\\", "/").strip("/")
    dirs, files = list_dir(cwd_rel)

    crumb = [html.Strong("workspace")]
    if cwd_rel:
        for part in cwd_rel.split("/"):
            crumb.append(html.Span(" / "))
            crumb.append(html.Span(part))

    rows = []
    if cwd_rel:
        rows.append(
            dbc.Button(
                "↑ 上级目录",
                id={"type": "fs-up", "x": "1"},
                className="fs-btn",
                n_clicks=0,
            )
        )

    for name in dirs:
        rows.append(
            dbc.Button(
                ["[dir]  ", name],
                id={"type": "fs-dir", "name": name},
                className="fs-btn",
                n_clicks=0,
            )
        )

    for name in files:
        rel = join_rel(cwd_rel, name)
        rows.append(
            dbc.Button(
                name,
                id={"type": "fs-file", "path": rel},
                className="fs-btn fs-file",
                n_clicks=0,
            )
        )

    if not rows:
        rows.append(html.Div("（空文件夹）", className="text-muted small p-2"))

    return html.Div([
        html.Div(crumb, className="fs-crumb"),
        html.Div(rows, style={"maxHeight": "26vh", "overflowY": "auto"}),
    ])


def open_full_link(filepath: str):
    rel = rel_workspace(filepath) if os.path.isabs(filepath) else filepath.replace("\\", "/")
    return html.A(
        "新窗口打开",
        href=f"/open/{quote(rel)}",
        target="_blank",
        rel="noopener noreferrer",
        className="btn btn-outline-secondary btn-sm",
    )


def preview_header(icon_title, filepath):
    return html.Div(
        [html.H5(icon_title, className="mb-0 text-truncate"), open_full_link(filepath)],
        className="d-flex justify-content-between align-items-center mb-3 gap-2",
    )


def build_chat_ui(history, trailing=None):
    """只展示用户话 + 最终回复。"""
    chat_ui = []
    for msg in history:
        msg_type = msg.get("role") if isinstance(msg, dict) else msg.type
        msg_content = msg.get("content") if isinstance(msg, dict) else msg.content
        if msg_type in ["user", "human"]:
            chat_ui.append(
                html.Div(
                    [
                        html.Div("你", className="msg-label"),
                        html.Div(msg_content, className="user-bubble"),
                    ],
                    className="msg-row",
                )
            )
            continue
        if msg_type not in ["assistant", "ai"] or not msg_content:
            continue
        tool_calls = (
            msg.get("tool_calls") if isinstance(msg, dict) else getattr(msg, "tool_calls", None)
        )
        if tool_calls:
            continue
        chat_ui.append(
            html.Div(
                [
                    html.Div("Agent", className="msg-label"),
                    dcc.Markdown(msg_content, className="ai-bubble"),
                ],
                className="msg-row",
            )
        )
    if trailing:
        chat_ui.append(trailing)
    if not chat_ui:
        chat_ui.append(
            html.Div(
                [
                    html.Div("从一条指令开始", className="msg-label"),
                    html.Div(
                        "清洗表格、出仓脱敏、查知识库，或打开左侧历史继续上次工作。",
                        className="ai-bubble",
                        style={"color": "var(--ink-soft)"},
                    ),
                ],
                className="msg-row",
            )
        )
    return chat_ui


def build_history_sidebar(active_label: str = "当前对话"):
    loaded = _nav.get("loaded_from")
    items = [
        dbc.Button(
            "+ 新对话",
            id="btn-new-session",
            color="primary",
            size="sm",
            className="w-100 mb-2",
            n_clicks=0,
        ),
        html.Div(active_label, className="hist-section"),
        dbc.Button(
            "进行中" if not loaded else "进行中草稿",
            id="btn-active-session",
            className="hist-open-btn" + ("" if loaded else " active-item"),
            n_clicks=0,
        ),
        html.Div("历史", className="hist-section"),
    ]
    archived = list_archived_sessions(WORKSPACE_DIR)
    if not archived:
        items.append(html.Div("暂无归档", className="text-muted small px-1"))
    for s in archived:
        label = s["topic"] or "未命名"
        sub = s["archived_at"] or ""
        is_on = loaded == s["filename"]
        items.append(
            html.Div(
                [
                    dbc.Button(
                        [
                            html.Div(label, className="hist-title text-truncate"),
                            html.Small(sub, className="text-muted"),
                        ],
                        id={"type": "hist-session", "file": s["filename"]},
                        className="hist-open-btn" + (" active-item" if is_on else ""),
                        n_clicks=0,
                    ),
                    html.Button(
                        "×",
                        id={"type": "hist-del", "file": s["filename"]},
                        className="hist-del-btn",
                        title="删除",
                        n_clicks=0,
                        type="button",
                    ),
                ],
                className="hist-row",
            )
        )
    return html.Div(items, id="history-list")


def _persist_active_unlocked():
    """调用方需已持有 _job_lock，或确认单线程安全。"""
    save_active_session(
        chat_history,
        conversation_summary,
        WORKSPACE_DIR,
        summary_covers=conversation_summary_covers,
    )


class _CancelCallback(BaseCallbackHandler):
    """在 LLM / Tool 边界检查中断标志。"""

    def __init__(self, cancel_event: threading.Event):
        super().__init__()
        self._cancel = cancel_event

    def _raise_if_cancelled(self):
        if self._cancel.is_set():
            raise InterruptedError("用户中断")

    def on_llm_start(self, *args, **kwargs):
        self._raise_if_cancelled()

    def on_tool_start(self, *args, **kwargs):
        self._raise_if_cancelled()


def _run_agent_job(
    full_history_snapshot,
    epoch=None,
    kb_scope=None,
    thinking=False,
):
    """后台执行：写时备份 + stream；验收按 Policy Context + policy_refs。"""
    global chat_history, conversation_summary, conversation_summary_covers
    cancel_event = threading.Event()
    begin_cow_session()  # 只建清单，不拷全量文件
    # 清掉上一轮交付声明，避免旧 manifest 误触发
    clear_compliance_manifest(WORKSPACE_DIR)
    set_kb_scope(kb_scope)
    think = bool(thinking)
    active_llm = get_llm(thinking=think)
    active_agent = get_agent(thinking=think)

    with _job_lock:
        _job["cancel"] = cancel_event
        _job["thinking"] = think
        summary_snap = conversation_summary
        covers_snap = conversation_summary_covers
        start_epoch = _session_epoch if epoch is None else epoch

    set_cancel_checker(lambda: cancel_event.is_set())
    finished_ok = False
    # 硬校验失败后最多再让 Agent 自动修复几轮（不含首轮）
    max_repair_rounds = 2

    def _stale() -> bool:
        return start_epoch != _session_epoch

    try:
        model_msgs, new_summary, new_covers = build_model_context(
            full_history_snapshot,
            summary_snap,
            active_llm,
            summary_covers=covers_snap,
        )
        if cancel_event.is_set() or _stale():
            raise InterruptedError("用户中断")

        # 注入勾选状态 + 目录；交付按 Policy Context 归属
        working_msgs = [
            {"role": "user", "content": build_kb_runtime_hint(kb_scope)},
            {"role": "assistant", "content": "收到，按本轮勾选与目录判断是否查库。"},
            {"role": "user", "content": build_compliance_runtime_hint()},
            {
                "role": "assistant",
                "content": (
                    "收到。有新文件将写 `.compliance_manifest.json`，"
                    "用 deliveries[].files + policy_refs 声明归属；"
                    "无 Policy 用 policy_refs=[]；假 ctx 会被打回。"
                ),
            },
        ] + list(model_msgs)
        collected_new = []
        validation_note = None

        for attempt in range(1 + max_repair_rounds):
            if cancel_event.is_set() or _stale():
                raise InterruptedError("用户中断")

            final_messages = None
            for state in active_agent.stream(
                {"messages": working_msgs},
                config={"callbacks": [_CancelCallback(cancel_event)]},
                stream_mode="values",
            ):
                if cancel_event.is_set() or _stale():
                    raise InterruptedError("用户中断")
                final_messages = state.get("messages", final_messages)

            if final_messages is None:
                raise RuntimeError("Agent 未返回任何消息")

            round_new = list(final_messages[len(working_msgs) :])
            collected_new.extend(round_new)

            cow = get_cow_session()
            new_files = cow.list_new_files() if cow is not None else []
            result = finalize_job_delivery(WORKSPACE_DIR, new_files=new_files)
            status = result.get("status") or ""
            if result.get("skipped") or status == "NO_FORMAL_DELIVERY":
                break

            user_msg = format_user_delivery_message(result)
            if result.get("ok"):
                if user_msg:
                    validation_note = user_msg
                break

            feedback = format_validation_feedback(result)
            if attempt >= max_repair_rounds:
                validation_note = user_msg or (
                    "⚠️ 交付验收仍未通过，本轮不视为合规/完整登记交付。\n" + feedback
                )
                collected_new.append({"role": "ai", "content": validation_note})
                validation_note = None
                break

            # 修复轮：附带刷新后的 ctx 快照（检索可能发生在首轮之后）
            repair_ctx = format_policy_contexts_hint()
            working_msgs = list(final_messages) + [
                {
                    "role": "user",
                    "content": feedback + "\n\n【刷新】" + repair_ctx,
                }
            ]

        if validation_note:
            collected_new.append({"role": "ai", "content": validation_note})

        with _job_lock:
            if _stale():
                _job["error"] = "stale"
                _job["running"] = False
            else:
                chat_history = list(full_history_snapshot) + list(collected_new)
                conversation_summary = new_summary
                conversation_summary_covers = new_covers
                _persist_active_unlocked()
                _job["error"] = None
                _job["running"] = False
                finished_ok = True
    except InterruptedError:
        request_kill_sandbox()
        restored, deleted_new = end_cow_session(rollback=True)
        tip = (
            f"⏹ 已中断：本轮思考已停止。"
            f"写时备份回退完成（恢复/写回 {restored} 个文件，删除本轮新建 {deleted_new} 个）。"
        )
        with _job_lock:
            if _stale():
                _job["error"] = "stale"
                _job["running"] = False
            else:
                chat_history = list(full_history_snapshot) + [{"role": "ai", "content": tip}]
                _persist_active_unlocked()
                _job["error"] = "cancelled"
                _job["running"] = False
    except Exception as e:
        with _job_lock:
            if _stale():
                _job["error"] = "stale"
                _job["running"] = False
            else:
                if chat_history and (
                    (isinstance(chat_history[-1], dict) and chat_history[-1].get("role") == "user")
                    or getattr(chat_history[-1], "type", None) == "human"
                ):
                    chat_history = chat_history[:-1]
                chat_history = list(chat_history) + [{"role": "ai", "content": f"❌ 系统崩溃: {e}"}]
                _persist_active_unlocked()
                _job["error"] = str(e)
                _job["running"] = False
    finally:
        set_cancel_checker(None)
        clear_kb_scope()
        if finished_ok:
            end_cow_session(rollback=False)
        elif _job.get("error") == "cancelled":
            pass  # 已在 InterruptedError 分支 rollback
        else:
            # 崩溃或 stale：清理临时 CoW 目录；崩溃不自动回退文件
            end_cow_session(rollback=False)


@app.server.route("/open/<path:filename>")
def open_workspace_file(filename):
    path = resolve_workspace_path(filename)
    if not os.path.isfile(path):
        abort(404)

    name = os.path.basename(path)
    ext = os.path.splitext(name)[1].lower()
    directory = os.path.dirname(path)

    if ext in _IMAGE_EXT or ext == ".html":
        return send_from_directory(directory, name)

    if ext in _TABLE_EXT:
        try:
            df = pd.read_csv(path) if ext == ".csv" else pd.read_excel(path)
            page = build_standalone_table_page(
                f"{rel_workspace(path)}（共 {len(df)} 行）",
                df,
            )
            return Response(page, mimetype="text/html; charset=utf-8")
        except Exception as e:
            return Response(f"读取失败: {e}", status=500)

    if ext in _TEXT_EXT:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{name}</title>
<style>body{{margin:24px;font-family:Consolas,monospace}} pre{{white-space:pre-wrap}}</style>
</head><body><h1>{rel_workspace(path)}</h1><pre>{text}</pre></body></html>"""
        return Response(page, mimetype="text/html; charset=utf-8")

    return send_from_directory(directory, name)


def serve_layout():
    """每次打开/刷新：归档上次草稿 → 空白新会话。"""
    rel = _archive_and_reset_to_blank(cancel_job=True)
    status = ""
    if rel:
        status = f"上次对话已归档 · {rel}"
    return html.Div(
        [
            dcc.Store(id="files-tick", data=0),
            dcc.Store(id="cwd-rel", data=""),
            dcc.Store(id="selected-file", data=None),
            dcc.Store(id="history-tick", data=0),
            dcc.Store(id="thinking-on", data=False),
            dcc.Store(id="pending-delete-file", data=None),
            dcc.Interval(id="agent-poll", interval=800, n_intervals=0, disabled=True),
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("删除历史会话")),
                    dbc.ModalBody(id="delete-modal-body", children="确定删除？"),
                    dbc.ModalFooter(
                        [
                            dbc.Button(
                                "取消",
                                id="btn-delete-cancel",
                                outline=True,
                                className="me-2",
                                n_clicks=0,
                            ),
                            dbc.Button(
                                "删除",
                                id="btn-delete-confirm",
                                color="danger",
                                n_clicks=0,
                            ),
                        ]
                    ),
                ],
                id="delete-modal",
                is_open=False,
                centered=True,
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H1(
                                ["Cyber", html.Span("Lobster")],
                                className="brand-name",
                            ),
                            html.Div("本地数据分析 Agent", className="brand-tag"),
                        ],
                        className="brand-bar",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Div(
                                    [
                                        html.Div("会话", className="panel-head"),
                                        html.Div(
                                            html.Div(
                                                id="history-sidebar",
                                                children=build_history_sidebar(),
                                            ),
                                            className="panel-body",
                                        ),
                                    ],
                                    className="panel",
                                ),
                                lg=2,
                                md=12,
                                className="mb-3",
                            ),
                            dbc.Col(
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Span("对话"),
                                                html.Span(
                                                    id="agent-status",
                                                    className="status-line mb-0",
                                                    style={"marginTop": 0},
                                                    children=status or "就绪",
                                                ),
                                            ],
                                            className="panel-head",
                                        ),
                                        html.Div(
                                            html.Div(
                                                id="chat-box",
                                                children=build_chat_ui([]),
                                                className="chat-scroll",
                                            ),
                                            className="panel-body",
                                            style={"padding": "10px"},
                                        ),
                                        html.Div(
                                            [
                                                dcc.Upload(
                                                    id="upload-files",
                                                    children=html.Div(
                                                        [
                                                            "拖拽或点击上传 → ",
                                                            html.Code("_inbox/"),
                                                        ]
                                                    ),
                                                    multiple=True,
                                                    className="upload-zone",
                                                ),
                                                html.Div(
                                                    id="upload-status",
                                                    className="small text-success mb-2",
                                                ),
                                                html.Div(
                                                    [
                                                        html.Label("知识库范围"),
                                                        dbc.Select(
                                                            id="kb-scope",
                                                            options=(
                                                                [
                                                                    {
                                                                        "label": "未选择（查库前会先问你）",
                                                                        "value": "__none__",
                                                                    },
                                                                    {
                                                                        "label": "全部知识库（已确认跨库）",
                                                                        "value": "__all__",
                                                                    },
                                                                ]
                                                                + [
                                                                    {
                                                                        "label": f"仅 {c}",
                                                                        "value": c,
                                                                    }
                                                                    for c in list_knowledge_categories()
                                                                ]
                                                            ),
                                                            value="__none__",
                                                            size="sm",
                                                        ),
                                                        html.Div(
                                                            className="small text-muted mt-1",
                                                            children=(
                                                                "勾选知识库后检索会生成 Policy Context；"
                                                                "有新文件须写 deliveries + policy_refs 做归属验收。"
                                                            ),
                                                        ),
                                                    ],
                                                    className="composer-field mb-2",
                                                ),
                                                html.Div(
                                                    [
                                                        dbc.Button(
                                                            "思考",
                                                            id="thinking-btn",
                                                            n_clicks=0,
                                                            size="sm",
                                                            className="thinking-btn me-2",
                                                            title="DeepSeek 思考模式",
                                                        ),
                                                        html.Span(
                                                            id="thinking-label",
                                                            className="small text-muted",
                                                            children="关",
                                                        ),
                                                    ],
                                                    className="mb-2",
                                                ),
                                                dbc.InputGroup(
                                                    [
                                                        dbc.Input(
                                                            id="user-input",
                                                            placeholder="输入分析指令…",
                                                            type="text",
                                                        ),
                                                        dbc.Button(
                                                            "发送",
                                                            id="send-btn",
                                                            color="primary",
                                                            n_clicks=0,
                                                        ),
                                                        dbc.Button(
                                                            "中断",
                                                            id="stop-btn",
                                                            color="danger",
                                                            outline=True,
                                                            n_clicks=0,
                                                            disabled=True,
                                                            title="停止并回退本轮文件改动",
                                                        ),
                                                    ]
                                                ),
                                            ],
                                            className="panel-foot",
                                        ),
                                    ],
                                    className="panel",
                                ),
                                lg=5,
                                md=12,
                                className="mb-3",
                            ),
                            dbc.Col(
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Span("工作区"),
                                                dbc.Button(
                                                    "刷新",
                                                    id="btn-refresh-files",
                                                    color="link",
                                                    size="sm",
                                                    className="py-0",
                                                ),
                                            ],
                                            className="panel-head",
                                        ),
                                        html.Div(
                                            [
                                                html.Div(id="dir-browser"),
                                                html.Hr(className="my-2"),
                                                html.Div(
                                                    id="file-preview-panel",
                                                    className="preview-empty",
                                                    children="选择文件以预览",
                                                    style={
                                                        "height": "38vh",
                                                        "overflowY": "auto",
                                                        "textAlign": "left",
                                                    },
                                                ),
                                            ],
                                            className="panel-body",
                                        ),
                                    ],
                                    className="panel",
                                ),
                                lg=5,
                                md=12,
                                className="mb-3",
                            ),
                        ],
                        className="g-3",
                    ),
                ],
                className="app-shell",
            ),
        ]
    )


app.layout = serve_layout


# ---------- 渲染当前目录列表 ----------
@app.callback(
    Output("dir-browser", "children"),
    [Input("cwd-rel", "data"),
     Input("files-tick", "data"),
     Input("btn-refresh-files", "n_clicks")],
)
def render_browser(cwd_rel, tick, n_refresh):
    return build_browser(cwd_rel or "")


# ---------- 进入子目录 ----------
@app.callback(
    Output("cwd-rel", "data", allow_duplicate=True),
    Input({"type": "fs-dir", "name": ALL}, "n_clicks"),
    State("cwd-rel", "data"),
    prevent_initial_call=True,
)
def enter_dir(n_clicks, cwd_rel):
    if not ctx.triggered_id or not any(n_clicks or []):
        return no_update
    name = ctx.triggered_id["name"]
    return join_rel(cwd_rel or "", name)


# ---------- 返回上级 ----------
@app.callback(
    Output("cwd-rel", "data", allow_duplicate=True),
    Input({"type": "fs-up", "x": ALL}, "n_clicks"),
    State("cwd-rel", "data"),
    prevent_initial_call=True,
)
def go_up(n_clicks, cwd_rel):
    if not ctx.triggered_id or not any(n_clicks or []):
        return no_update
    cwd = (cwd_rel or "").replace("\\", "/").strip("/")
    if not cwd:
        return ""
    parent = "/".join(cwd.split("/")[:-1])
    return parent


# ---------- 点击文件 → 预览 ----------
@app.callback(
    Output("selected-file", "data"),
    Input({"type": "fs-file", "path": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def select_file(n_clicks):
    if not ctx.triggered_id or not any(n_clicks or []):
        return no_update
    rel = ctx.triggered_id["path"]
    path = resolve_workspace_path(rel)
    if not os.path.isfile(path):
        return no_update
    return path


# ---------- 上传到 _inbox ----------
@app.callback(
    [Output("upload-status", "children"),
     Output("files-tick", "data", allow_duplicate=True),
     Output("cwd-rel", "data", allow_duplicate=True)],
    Input("upload-files", "contents"),
    [State("upload-files", "filename"),
     State("files-tick", "data")],
    prevent_initial_call=True,
)
def handle_upload(contents_list, filenames, files_tick):
    if not contents_list:
        return no_update, no_update, no_update
    if isinstance(contents_list, str):
        contents_list = [contents_list]
        filenames = [filenames]

    saved = []
    for content, name in zip(contents_list, filenames or []):
        if not name:
            continue
        safe_name = os.path.basename(name)
        try:
            _header, b64data = content.split(",", 1)
            dest = os.path.join(INBOX_DIR, safe_name)
            with open(dest, "wb") as f:
                f.write(base64.b64decode(b64data))
            saved.append(safe_name)
        except Exception as e:
            return f"❌ 上传失败 ({safe_name}): {e}", no_update, no_update

    if not saved:
        return "未保存任何文件", no_update, no_update
    tip = f"已上传 {len(saved)} 个文件到 _inbox/。可让 Agent 按主题归类。"
    return tip, (files_tick or 0) + 1, "_inbox"


# ---------- 思考开关 ----------
@app.callback(
    [
        Output("thinking-on", "data"),
        Output("thinking-btn", "className"),
        Output("thinking-label", "children"),
    ],
    Input("thinking-btn", "n_clicks"),
    State("thinking-on", "data"),
    prevent_initial_call=True,
)
def toggle_thinking(n_clicks, on):
    if not n_clicks:
        return no_update, no_update, no_update
    nxt = not bool(on)
    cls = "thinking-btn me-2 on" if nxt else "thinking-btn me-2"
    return nxt, cls, ("开" if nxt else "关")


@app.callback(
    Output("history-sidebar", "children"),
    Input("history-tick", "data"),
)
def refresh_history_sidebar(_tick):
    return build_history_sidebar()


# ---------- 新会话 / 历史 ----------
@app.callback(
    [Output("chat-box", "children", allow_duplicate=True),
     Output("agent-status", "children", allow_duplicate=True),
     Output("files-tick", "data", allow_duplicate=True),
     Output("history-tick", "data", allow_duplicate=True)],
    Input("btn-new-session", "n_clicks"),
    [State("files-tick", "data"), State("history-tick", "data")],
    prevent_initial_call=True,
)
def new_session_click(n_clicks, files_tick, hist_tick):
    if not n_clicks:
        return no_update, no_update, no_update, no_update
    rel = _archive_and_reset_to_blank(cancel_job=True)
    tip = f"已开启新会话。旧对话已归档到 {rel}" if rel else "已开启新会话。"
    return [], tip, (files_tick or 0) + 1, (hist_tick or 0) + 1


@app.callback(
    [Output("chat-box", "children", allow_duplicate=True),
     Output("agent-status", "children", allow_duplicate=True),
     Output("history-tick", "data", allow_duplicate=True)],
    Input("btn-active-session", "n_clicks"),
    State("history-tick", "data"),
    prevent_initial_call=True,
)
def focus_active_draft(n_clicks, tick):
    global chat_history, conversation_summary, conversation_summary_covers
    if not n_clicks:
        return no_update, no_update, no_update
    with _job_lock:
        if _job["running"]:
            return no_update, "Agent 运行中，请先中断", no_update
        if _nav.get("loaded_from"):
            chat_history, conversation_summary, conversation_summary_covers = (
                load_active_session(WORKSPACE_DIR)
            )
        _nav["loaded_from"] = None
        history = list(chat_history)
    return build_chat_ui(history), "当前进行中", (tick or 0) + 1


@app.callback(
    [Output("chat-box", "children", allow_duplicate=True),
     Output("agent-status", "children", allow_duplicate=True),
     Output("history-tick", "data", allow_duplicate=True)],
    Input({"type": "hist-session", "file": ALL}, "n_clicks"),
    State("history-tick", "data"),
    prevent_initial_call=True,
)
def open_history_session(n_clicks, tick):
    global chat_history, conversation_summary, conversation_summary_covers
    if not isinstance(ctx.triggered_id, dict):
        return no_update, no_update, no_update
    if ctx.triggered_id.get("type") != "hist-session":
        return no_update, no_update, no_update
    if not any(n_clicks or []):
        return no_update, no_update, no_update
    filename = ctx.triggered_id["file"]
    if filename == _nav.get("loaded_from"):
        return no_update, no_update, no_update
    with _job_lock:
        if _job["running"]:
            return no_update, "Agent 运行中，请先中断", no_update
        history, summary, topic, covers = load_archived_session(WORKSPACE_DIR, filename)
        if not history and not summary:
            return no_update, f"无法打开：{filename}", no_update
        if not _nav.get("loaded_from"):
            _persist_active_unlocked()
        chat_history = history
        conversation_summary = summary
        conversation_summary_covers = covers
        _nav["loaded_from"] = filename
    return build_chat_ui(history), f"已打开历史：{topic}", (tick or 0) + 1


@app.callback(
    [Output("delete-modal", "is_open"),
     Output("pending-delete-file", "data"),
     Output("delete-modal-body", "children")],
    Input({"type": "hist-del", "file": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def ask_delete_history(n_clicks):
    if not isinstance(ctx.triggered_id, dict):
        return no_update, no_update, no_update
    if ctx.triggered_id.get("type") != "hist-del":
        return no_update, no_update, no_update
    if not any(n_clicks or []):
        return no_update, no_update, no_update
    filename = ctx.triggered_id["file"]
    topic = "该会话"
    for s in list_archived_sessions(WORKSPACE_DIR):
        if s["filename"] == filename:
            topic = s.get("topic") or topic
            break
    return True, filename, f"确定删除「{topic}」吗？此操作不可恢复。"


@app.callback(
    [Output("delete-modal", "is_open", allow_duplicate=True),
     Output("pending-delete-file", "data", allow_duplicate=True),
     Output("history-tick", "data", allow_duplicate=True),
     Output("agent-status", "children", allow_duplicate=True)],
    [Input("btn-delete-cancel", "n_clicks"),
     Input("btn-delete-confirm", "n_clicks")],
    State("pending-delete-file", "data"),
    State("history-tick", "data"),
    prevent_initial_call=True,
)
def resolve_delete_history(n_cancel, n_confirm, pending, tick):
    tid = ctx.triggered_id
    if tid == "btn-delete-cancel":
        return False, None, no_update, no_update
    if tid != "btn-delete-confirm" or not pending:
        return no_update, no_update, no_update, no_update
    ok = delete_archived_session(WORKSPACE_DIR, pending)
    if _nav.get("loaded_from") == pending:
        _nav["loaded_from"] = None
    tip = f"已删除 {pending}" if ok else f"删除失败：{pending}"
    return False, None, (tick or 0) + 1, tip


# ---------- 聊天 ----------
@app.callback(
    [Output("chat-box", "children", allow_duplicate=True),
     Output("user-input", "value"),
     Output("agent-status", "children", allow_duplicate=True),
     Output("agent-poll", "disabled", allow_duplicate=True),
     Output("send-btn", "disabled", allow_duplicate=True),
     Output("stop-btn", "disabled", allow_duplicate=True),
     Output("files-tick", "data", allow_duplicate=True),
     Output("thinking-btn", "disabled", allow_duplicate=True),
     Output("history-tick", "data", allow_duplicate=True)],
    [Input("send-btn", "n_clicks"),
     Input("user-input", "n_submit")],
    [State("user-input", "value"),
     State("files-tick", "data"),
     State("kb-scope", "value"),
     State("thinking-on", "data"),
     State("history-tick", "data")],
    prevent_initial_call=True,
)
def start_chat(
    n_clicks, n_submit, user_text, files_tick, kb_scope, thinking_on, hist_tick
):
    global chat_history, conversation_summary
    if not user_text:
        return (no_update,) * 9

    status_extra = ""
    files_tick_out = no_update
    hist_tick_out = no_update
    think = bool(thinking_on)

    # 用户说「新任务/清空…」→ 先归档再开空窗
    if wants_new_session(user_text):
        rel = _archive_and_reset_to_blank(cancel_job=True)
        status_extra = f"已归档到 {rel}。" if rel else "已清空会话。"
        files_tick_out = (files_tick or 0) + 1
        hist_tick_out = (hist_tick or 0) + 1
        user_text = strip_clear_keywords(user_text)
        if not user_text:
            return (
                [],
                "",
                f"✅ 新会话已开始。{status_extra}",
                True,
                False,
                True,
                files_tick_out,
                False,
                hist_tick_out,
            )

    with _job_lock:
        if _job["running"]:
            return (no_update, no_update, "⏳ Agent 仍在处理上一轮，可点「中断」停止",
                    no_update, no_update, no_update, no_update, no_update, no_update)
        # 从历史浏览发消息：回到草稿模式（继续写当前草稿上下文）
        if _nav.get("loaded_from"):
            _nav["loaded_from"] = None
        _job["running"] = True
        _job["error"] = None
        _job["cancel"] = threading.Event()
        _job["thinking"] = think
        chat_history = list(chat_history) + [{"role": "user", "content": user_text}]
        snapshot = list(chat_history)
        epoch = int(_session_epoch or 0)
        _persist_active_unlocked()

    threading.Thread(
        target=_run_agent_job,
        args=(snapshot, epoch, kb_scope, think),
        daemon=True,
    ).start()
    kb_tip = ""
    if not kb_scope or kb_scope == "__none__":
        kb_tip = "知识库未勾选 · "
    elif kb_scope == "__all__":
        kb_tip = "知识库：全部 · "
    else:
        kb_tip = f"知识库「{kb_scope}」· "
    deliv_tip = "验收：policy_refs · "
    mode = model_label(thinking=think)
    waiting = html.Div(
        f"⏳ {mode} 处理中…（{kb_tip}{deliv_tip}可点「中断」回退本轮文件）",
        className="mb-3 text-secondary",
    )
    status = f"🔄 运行中 · {mode}；{kb_tip}{deliv_tip}可中断"
    if status_extra:
        status = status_extra + " " + status
    return (
        build_chat_ui(snapshot, trailing=waiting),
        "",
        status,
        False,   # poll on
        True,    # send disabled
        False,   # stop enabled
        files_tick_out,
        True,    # thinking disabled while running
        hist_tick_out,
    )


@app.callback(
    Output("agent-status", "children", allow_duplicate=True),
    Input("stop-btn", "n_clicks"),
    prevent_initial_call=True,
)
def stop_agent(n_clicks):
    if not n_clicks:
        return no_update
    with _job_lock:
        if not _job["running"]:
            return "当前没有运行中的任务"
        _job["cancel"].set()
    request_kill_sandbox()
    return "⏹ 正在中断：停止思考，并回退本轮工作区文件改动…"


@app.callback(
    [Output("chat-box", "children"),
     Output("agent-status", "children"),
     Output("agent-poll", "disabled"),
     Output("send-btn", "disabled"),
     Output("stop-btn", "disabled"),
     Output("files-tick", "data"),
     Output("thinking-btn", "disabled"),
     Output("history-tick", "data")],
    Input("agent-poll", "n_intervals"),
    [State("files-tick", "data"), State("history-tick", "data")],
    prevent_initial_call=True,
)
def poll_agent(n_intervals, files_tick, hist_tick):
    with _job_lock:
        running = _job["running"]
        history = list(chat_history)
        err = _job.get("error")
        think = bool(_job.get("thinking"))

    if running:
        mode = model_label(thinking=think)
        waiting = html.Div(
            f"⏳ {mode} 处理中…（可点「中断」回退本轮文件）",
            className="mb-3 text-secondary",
        )
        return (
            build_chat_ui(history, trailing=waiting),
            f"🔄 运行中 · {mode}",
            False,
            True,
            False,
            no_update,
            True,
            no_update,
        )

    status = "✅ 本轮完成"
    if err == "cancelled":
        status = "⏹ 已中断，本轮文件改动已回退"
    elif err:
        status = f"⚠️ 本轮异常结束：{err}"
    return (
        build_chat_ui(history),
        status,
        True,
        False,
        True,
        (files_tick or 0) + 1,
        False,
        (hist_tick or 0) + 1,
    )


# ---------- 预览 ----------
@app.callback(
    Output("file-preview-panel", "children"),
    Input("selected-file", "data"),
)
def render_selected_file(filepath):
    if not filepath or not os.path.exists(filepath):
        return html.Div(
            "选择目录中的文件以预览",
            className="preview-empty",
        )

    name = os.path.basename(filepath)
    rel = rel_workspace(filepath)
    ext = os.path.splitext(name)[1].lower()

    if ext == ".html":
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return html.Div([
            preview_header(f"📊 {rel}", filepath),
            html.Iframe(srcDoc=content, style={"width": "100%", "height": "320px", "border": "none"}),
        ])

    if ext in _TABLE_EXT:
        try:
            df = pd.read_csv(filepath) if ext == ".csv" else pd.read_excel(filepath)
            total = len(df)
            shown = df
            tip = f"共 {total} 行 · 可排序 / 列筛选 / 翻页"
            if total > _TABLE_PREVIEW_MAX_ROWS:
                shown = df.head(_TABLE_PREVIEW_MAX_ROWS)
                tip = (
                    f"面板预览前 {_TABLE_PREVIEW_MAX_ROWS} 行（共 {total} 行）。"
                    f"点右上角「新窗口」可对全表排序筛选。"
                )
            return html.Div([
                preview_header(f"📁 {rel}", filepath),
                html.P(tip, className="text-muted small"),
                build_dash_datatable(shown, page_size=12),
            ])
        except Exception as e:
            return html.Div(f"表格读取失败: {e}", className="text-danger")

    if ext in _IMAGE_EXT:
        mime = {
            ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".gif": "image/gif", ".webp": "image/webp", ".svg": "image/svg+xml",
        }[ext]
        with open(filepath, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return html.Div([
            preview_header(f"🖼️ {rel}", filepath),
            html.Img(
                src=f"data:{mime};base64,{b64}",
                style={"maxWidth": "100%", "maxHeight": "32vh", "height": "auto", "objectFit": "contain"},
            ),
        ])

    if ext in _TEXT_EXT:
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                text = f.read(8000)
            return html.Div([
                preview_header(f"📄 {rel}", filepath),
                html.Pre(text, style={
                    "whiteSpace": "pre-wrap", "background": "#f8f9fa",
                    "padding": "12px", "borderRadius": "6px", "fontSize": "13px",
                }),
            ])
        except Exception as e:
            return html.Div(f"文本读取失败: {e}", className="text-danger")

    size_kb = os.path.getsize(filepath) / 1024
    return html.Div([
        preview_header(f"📄 {rel}", filepath),
        html.P(f"类型：{ext or '无扩展名'} · {size_kb:.1f} KB"),
        html.P("可尝试在新窗口打开。", className="text-muted"),
    ])


if __name__ == "__main__":
    # use_reloader=False：避免 Dash 热重载弄坏 chromadb Rust 绑定
    # （典型报错：'RustBindingsAPI' object has no attribute 'bindings'）
    app.run(debug=True, port=8050, threaded=True, use_reloader=False)
