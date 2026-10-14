import os
import subprocess
import sys
import threading
import shutil
from pathlib import Path
from typing import Callable, Optional

from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from cow_backup import CowSession, build_sandbox_cow_preamble
from job_policy_state import (
    append_policy_contexts_for_categories,
    begin_rag_call,
    clear_policy_contexts,
)
from parent_store import get_parent, load_parent_store

# 国内访问 HuggingFace 不稳定时走镜像（可被环境变量覆盖）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 路径一律相对脚本位置，不依赖终端 cwd
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # cyber_lobster_project

# 工作区放在本项目目录内：cyber_lobster_project/workspace
WORKSPACE_DIR = os.path.join(_BASE_DIR, "workspace")
os.makedirs(WORKSPACE_DIR, exist_ok=True)
RAW_DIR = os.path.join(WORKSPACE_DIR, "raw")
os.makedirs(RAW_DIR, exist_ok=True)

_KNOWLEDGE_DIR = os.path.join(_BASE_DIR, "knowledge_docs")
_CHROMA_PATH = os.path.join(_BASE_DIR, "chroma_db")

def list_knowledge_categories() -> list[str]:
    """knowledge_docs 下一级子文件夹 = 可选知识库分类（每次读盘，便于新增后刷新）。"""
    if not os.path.isdir(_KNOWLEDGE_DIR):
        return []
    return sorted(
        f
        for f in os.listdir(_KNOWLEDGE_DIR)
        if os.path.isdir(os.path.join(_KNOWLEDGE_DIR, f))
        and not f.startswith(".")
    )


def format_knowledge_catalog(intro_chars: int = 120) -> str:
    """只列分类 / 文档标题 / 首行简介，不含正文。供注入上下文与 browse 工具，不要求勾选。"""
    cats = list_knowledge_categories()
    if not cats:
        return "（knowledge_docs 下暂无分类文件夹）"
    lines: list[str] = []
    for cat in cats:
        cat_path = os.path.join(_KNOWLEDGE_DIR, cat)
        files = sorted(
            f
            for f in os.listdir(cat_path)
            if os.path.isfile(os.path.join(cat_path, f))
            and not f.startswith(".")
            and f.lower().endswith((".txt", ".md", ".markdown", ".rst"))
        )
        if not files:
            lines.append(f"- [{cat}] （空目录）")
            continue
        for name in files:
            path = os.path.join(cat_path, name)
            title = os.path.splitext(name)[0]
            intro = ""
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        s = line.strip()
                        if s:
                            intro = s[:intro_chars]
                            if len(s) > intro_chars:
                                intro += "…"
                            break
            except OSError:
                intro = "（无法读取简介）"
            lines.append(f"- [{cat}] {title}: {intro or '（无简介）'}")
    return "\n".join(lines)


def describe_kb_scope(scope: Optional[str] = None) -> str:
    """把界面勾选值说成人话。"""
    if scope is None:
        scope = get_kb_scope()
    if scope is None or scope == "__none__":
        return "未选择（正文检索未授权）"
    if scope == "__all__":
        return "全部知识库（用户已确认可跨库）"
    return f"仅 {scope}（用户已确认，硬锁定）"


def build_kb_runtime_hint(kb_scope_raw: Optional[str] = None) -> str:
    """每轮注入给模型：勾选状态 + 目录标题，避免瞎问勾选、避免先撞检索。"""
    # 与 set_kb_scope 一致地规范化展示
    raw = (kb_scope_raw or "").strip() or "__none__"
    return (
        "【系统注入 · 本轮知识库状态】（不是用户原话，请遵守）\n"
        f"界面勾选: {describe_kb_scope(raw)}\n"
        "本地知识库目录（仅标题/简介，用于判断「像不像有关」；不是规范正文）：\n"
        f"{format_knowledge_catalog()}\n"
        "行为：闲聊/普通数据分析勿提勾选；仅当目录标题或简介与当前任务明显相关时才考虑查库；"
        "若相关且仍为「未选择」，先简短请用户勾选后再继续；若已勾选，可直接 search_knowledge_base；"
        "目录明显无关则不要问勾选，按通用方式回答并区分非库内知识。"
    )


_valid_categories = list_knowledge_categories()
_cat_str = (
    ", ".join([f"'{c}'" for c in _valid_categories]) if _valid_categories else "无特定分类"
)

# Embedding / Chroma：懒加载 + 单例 + 锁，避免 debug 热重载 / 并发把 Rust 绑定打坏
_embeddings = None
_vectorstore = None
_chroma_lock = threading.Lock()

# 中断支持：外部可注入「是否已取消」；沙箱子进程 / 容器可被强制杀掉
_cancel_checker: Optional[Callable[[], bool]] = None
_active_proc = None
_active_container_id: Optional[str] = None
_proc_lock = threading.Lock()
_cow_session: Optional[CowSession] = None
_cow_lock = threading.Lock()

# 沙箱后端：auto（有 Docker 用 Docker，否则本地）| docker | local
# 环境变量 SANDBOX_BACKEND；Docker 镜像 SANDBOX_DOCKER_IMAGE（默认 cyberlobster-sandbox:py311）
_SANDBOX_IMAGE = (os.getenv("SANDBOX_DOCKER_IMAGE") or "cyberlobster-sandbox:py311").strip()
_SANDBOX_BACKEND = (os.getenv("SANDBOX_BACKEND") or "auto").strip().lower()


def _docker_cli() -> Optional[str]:
    return shutil.which("docker")


def _docker_available() -> bool:
    cli = _docker_cli()
    if not cli:
        return False
    try:
        r = subprocess.run(
            [cli, "info"],
            capture_output=True,
            text=True,
            timeout=8,
        )
        return r.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def resolve_sandbox_backend() -> str:
    """返回实际使用的后端：docker | local。每次读取环境变量，便于 .env 生效。"""
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    except Exception:
        pass
    mode = (os.getenv("SANDBOX_BACKEND") or _SANDBOX_BACKEND or "auto").strip().lower()
    if mode in ("local", "subprocess", "host"):
        return "local"
    if mode == "docker":
        if _docker_available():
            return "docker"
        return "local"  # 强制 docker 但不可用时降级，避免整单挂死
    # auto
    return "docker" if _docker_available() else "local"


def sandbox_backend_label() -> str:
    b = resolve_sandbox_backend()
    if b == "docker":
        return f"docker ({_SANDBOX_IMAGE})"
    return "local-subprocess"


def _workspace_mount_spec() -> str:
    """Docker -v 左侧路径：Windows 下用 resolve 绝对路径，Docker Desktop 可挂载。"""
    host = str(Path(WORKSPACE_DIR).resolve())
    # Docker Desktop on Windows accepts D:\path or //d/path; prefer resolved native path
    return f"{host}:/workspace"

# UI 硬路由：由用户勾选；__none__=未选（禁止检索）；__all__=用户确认全库；其他=单库硬锁
_kb_scope: Optional[str] = None
_kb_scope_lock = threading.Lock()


def set_cancel_checker(fn: Optional[Callable[[], bool]]) -> None:
    global _cancel_checker
    _cancel_checker = fn


def set_kb_scope(category: Optional[str]) -> None:
    """本轮用户确认的知识库范围。任务结束应清回 None。"""
    global _kb_scope
    raw = (category or "").strip() or "__none__"
    with _kb_scope_lock:
        _kb_scope = raw


def get_kb_scope() -> Optional[str]:
    with _kb_scope_lock:
        return _kb_scope


def clear_kb_scope() -> None:
    global _kb_scope
    with _kb_scope_lock:
        _kb_scope = None
    # Job 结束时常与 clear_kb_scope 同路；Policy Context 在 begin_cow 已清，
    # 这里再清一次避免残留跨 Job。
    clear_policy_contexts()


def begin_cow_session() -> CowSession:
    """开跑时调用：只建清单，不拷文件；同时清空本轮 Policy Context。"""
    global _cow_session
    clear_policy_contexts()
    with _cow_lock:
        if _cow_session is not None:
            _cow_session.cleanup()
        _cow_session = CowSession(WORKSPACE_DIR)
        return _cow_session


def get_cow_session() -> Optional[CowSession]:
    with _cow_lock:
        return _cow_session


def end_cow_session(rollback: bool = False) -> tuple[int, int]:
    """结束本轮：可选回退，并清理临时备份目录。"""
    global _cow_session
    with _cow_lock:
        session = _cow_session
        _cow_session = None
    if session is None:
        return 0, 0
    restored = deleted = 0
    try:
        if rollback:
            restored, deleted = session.rollback()
    finally:
        session.cleanup()
    return restored, deleted


def _is_cancelled() -> bool:
    return bool(_cancel_checker and _cancel_checker())


def request_kill_sandbox() -> None:
    """中断时杀掉正在跑的沙箱子进程或 Docker 容器。"""
    with _proc_lock:
        proc = _active_proc
        cid = _active_container_id
    if cid:
        cli = _docker_cli()
        if cli:
            try:
                subprocess.run([cli, "kill", cid], capture_output=True, timeout=8)
            except (OSError, subprocess.TimeoutExpired):
                pass
    if proc is not None and proc.poll() is None:
        try:
            proc.kill()
        except OSError:
            pass


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embeddings


def _reset_chroma_clients() -> None:
    global _embeddings, _vectorstore
    _vectorstore = None
    _embeddings = None


def _get_vectorstore():
    """复用 Chroma 客户端；遇到 RustBindings 异常时重建一次。"""
    global _vectorstore
    with _chroma_lock:
        try:
            if _vectorstore is None:
                _vectorstore = Chroma(
                    persist_directory=_CHROMA_PATH,
                    embedding_function=_get_embeddings(),
                )
            return _vectorstore
        except AttributeError:
            # 典型：'RustBindingsAPI' object has no attribute 'bindings'
            _reset_chroma_clients()
            _vectorstore = Chroma(
                persist_directory=_CHROMA_PATH,
                embedding_function=_get_embeddings(),
            )
            return _vectorstore


@tool
def execute_python_code(code: str) -> str:
    """
    在隔离沙箱中执行 Python 代码（优先 Docker：仅挂载 workspace、禁网；不可用时回退本机子进程）。
    请将所有生成的数据表或图表保存到当前工作目录（workspace）下。
    如果代码运行报错，工具会返回详尽的 stderr，请根据报错修改代码并重新执行。
    """
    if _is_cancelled():
        return "❌ 任务已中断，跳过代码执行。"

    script_name = "temp_sandbox_script.py"
    script_path = os.path.join(WORKSPACE_DIR, script_name)
    session = get_cow_session()
    if session is not None:
        preamble = build_sandbox_cow_preamble(
            session.workspace_dir,
            session.backup_root,
            session.initial_files,
        )
        script_body = preamble + "\n" + code
    else:
        script_body = code

    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_body)

    backend = resolve_sandbox_backend()
    try:
        if backend == "docker":
            return _execute_in_docker(script_name)
        return _execute_local_subprocess(script_path)
    except Exception as e:
        return f"❌ 沙箱系统内部防御性拦截: {e}"
    finally:
        if os.path.exists(script_path):
            try:
                os.remove(script_path)
            except OSError:
                pass


def _execute_local_subprocess(script_path: str) -> str:
    proc = None
    try:
        proc = subprocess.Popen(
            [sys.executable, script_path],
            cwd=WORKSPACE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        with _proc_lock:
            global _active_proc
            _active_proc = proc

        try:
            stdout, stderr = proc.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            return (
                "❌ 运行超时（超过 15 秒）！请检查代码中是否存在 while True 死循环，"
                "或尝试优化处理逻辑。"
            )

        if _is_cancelled() or proc.returncode < 0:
            return "❌ 任务已中断，代码执行被终止。"

        if proc.returncode == 0:
            return f"✅ 代码执行成功！[local]\n标准输出 (stdout):\n{stdout}"
        return (
            f"❌ 代码执行失败！[local] (请分析报错并修复代码)\n"
            f"错误追踪 (stderr):\n{stderr}\n"
            f"标准输出:\n{stdout}"
        )
    finally:
        with _proc_lock:
            if _active_proc is proc:
                _active_proc = None


def _execute_in_docker(script_name: str) -> str:
    """仅挂载 workspace → /workspace，--network none，超时杀容器。"""
    cli = _docker_cli()
    if not cli:
        return _execute_local_subprocess(os.path.join(WORKSPACE_DIR, script_name))

    # 先 create 拿 container id，便于中断 kill；再 start -a 附着输出
    create_cmd = [
        cli,
        "create",
        "--network",
        "none",
        "-w",
        "/workspace",
        "-v",
        _workspace_mount_spec(),
        _SANDBOX_IMAGE,
        "python",
        f"/workspace/{script_name}",
    ]
    try:
        cr = subprocess.run(create_cmd, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return "❌ Docker 创建容器超时。可设置 SANDBOX_BACKEND=local 回退本机沙箱。"

    if cr.returncode != 0:
        err = (cr.stderr or cr.stdout or "").strip()
        # 常见：镜像不存在 → 回退本地并提示
        hint = (
            f"Docker 不可用或镜像缺失（{err[:400]}）。"
            f"已回退本机子进程。构建镜像：docker build -t {_SANDBOX_IMAGE} -f "
            f"cyber_lobster_project/docker/Dockerfile.sandbox cyber_lobster_project/docker"
        )
        local_out = _execute_local_subprocess(os.path.join(WORKSPACE_DIR, script_name))
        return f"⚠️ {hint}\n\n{local_out}"

    cid = (cr.stdout or "").strip()
    if not cid:
        return "❌ Docker create 未返回容器 ID，已中止。"

    with _proc_lock:
        global _active_container_id
        _active_container_id = cid

    try:
        start = subprocess.Popen(
            [cli, "start", "-a", cid],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        with _proc_lock:
            global _active_proc
            _active_proc = start
        try:
            stdout, stderr = start.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                subprocess.run([cli, "kill", cid], capture_output=True, timeout=8)
            except (OSError, subprocess.TimeoutExpired):
                pass
            start.kill()
            stdout, stderr = start.communicate()
            return (
                "❌ 运行超时（超过 15 秒）！[docker] 请检查死循环或优化逻辑。"
            )

        if _is_cancelled():
            return "❌ 任务已中断，代码执行被终止。[docker]"

        # docker start -a 的返回码即容器进程码
        if start.returncode == 0:
            return f"✅ 代码执行成功！[docker·禁网·仅 workspace]\n标准输出 (stdout):\n{stdout}"
        return (
            f"❌ 代码执行失败！[docker] (请分析报错并修复代码)\n"
            f"错误追踪 (stderr):\n{stderr}\n"
            f"标准输出:\n{stdout}"
        )
    finally:
        with _proc_lock:
            _active_container_id = None
            if _active_proc is not None:
                _active_proc = None
        try:
            subprocess.run([cli, "rm", "-f", cid], capture_output=True, timeout=8)
        except (OSError, subprocess.TimeoutExpired):
            pass


@tool(
    description=(
        "浏览本地知识库「目录」（分类、文档标题、首行简介），不返回规范正文。"
        "不要求用户勾选知识库。用于判断当前问题是否可能与某份文档有关；"
        "确认有关且用户已勾选后，再用 search_knowledge_base 取正文。"
        "闲聊或目录明显无关时不要调用。"
    )
)
def browse_knowledge_catalog() -> str:
    """标题级目录，无需 kb_scope。"""
    if _is_cancelled():
        return "❌ 任务已中断，跳过目录浏览。"
    if not os.path.isdir(_KNOWLEDGE_DIR):
        return "尚未找到 knowledge_docs 目录。"
    body = format_knowledge_catalog()
    return (
        "知识库目录（仅标题/简介，非正文；浏览不需要勾选）：\n"
        f"{body}\n"
        f"当前界面勾选: {describe_kb_scope()}"
    )


@tool(
    description=(
        "查询本地知识库正文（ParentDocument：子块定位 → 父块交付）。"
        "参数 query: 搜索关键词或问题。"
        "必须先由用户在界面勾选知识库范围；你不能自行定库。"
        "调用前应已根据目录标题/简介判断与任务相关，且界面非「未选择」。"
        "若返回「需要用户确认」，停止检索，简短请用户勾选后继续；"
        "禁止对无关闲聊反复要求勾选；禁止用训练知识冒充库内规范。"
    )
)
def search_knowledge_base(query: str, category: Optional[str] = None) -> str:
    """子块向量检索 → 按 parent_id 取父块 → 同一父块去重后交付。category 参数已废弃，忽略。"""
    if _is_cancelled():
        return "❌ 任务已中断，跳过知识库检索。"

    if not os.path.exists(_CHROMA_PATH):
        return "本地知识库尚未建立，请提醒用户先运行建库脚本。"

    scope = get_kb_scope()
    # 用户未在界面勾选 → 拒绝检索，逼 Agent 去问用户
    if scope is None or scope == "__none__":
        cats = "、".join(list_knowledge_categories()) or "（暂无分类文件夹）"
        return (
            "【需要用户确认知识库】界面当前为「未选择」，系统禁止自行定库或全库搜索。"
            f"请用一两句话询问用户：是否需要检索本地知识库？若需要，请在聊天框上方「知识库范围」"
            f"勾选具体分类（可选: {cats}）或「全部知识库」，然后回复「继续」等即可。"
            "在用户勾选并再次发送之前，不要调用本工具，也不要用训练知识冒充库内规范。"
        )

    # 用户已确认：单库硬锁，或全部（无 filter）
    if scope == "__all__":
        effective = None
        scope_label = "全部知识库（用户已确认）"
    else:
        effective = scope
        scope_label = f"仅 {scope}（用户已确认，禁止串库）"

    def _search(vs, filt):
        return vs.similarity_search(query, k=6, filter=filt)

    try:
        vectorstore = _get_vectorstore()
        filter_dict = {"category": effective} if effective else None
        child_hits = _search(vectorstore, filter_dict)
        # 用户已定范围：禁止 Fallback 串库
    except AttributeError:
        _reset_chroma_clients()
        try:
            vectorstore = _get_vectorstore()
            filter_dict = {"category": effective} if effective else None
            child_hits = _search(vectorstore, filter_dict)
        except Exception as e2:
            return f"知识库暂时不可用（{e2}）。请重启 app.py（建议关闭热重载）后再试。"
    except Exception as e:
        return f"知识库检索失败: {e}"

    if not child_hits:
        return (
            f"在用户确认的范围「{scope_label}」中未找到相关信息。"
            "请换关键词，或请用户改选知识库后再试；禁止改查未授权范围。"
        )

    parents = load_parent_store()
    seen_parent_ids = set()
    delivered = []
    hit_categories: list[str] = []

    for child in child_hits:
        parent_id = child.metadata.get("parent_id")
        if not parent_id or parent_id in seen_parent_ids:
            continue
        seen_parent_ids.add(parent_id)

        parent = parents.get(parent_id) or get_parent(parent_id)
        if parent:
            cat = parent.get("category", child.metadata.get("category", "通用"))
            src = parent.get("source", child.metadata.get("source", ""))
            body = parent.get("content", "")
            mode = "父块"
        else:
            cat = child.metadata.get("category", "通用")
            src = child.metadata.get("source", "")
            body = child.page_content
            mode = "子块(无父库)"

        hit_categories.append(str(cat))
        delivered.append(
            f"【知识片段 {len(delivered)+1} ({mode} | 领域: {cat} | 来源: {src} | parent_id: {parent_id})】:\n"
            f"{body}"
        )
        if len(delivered) >= 3:
            break

    if not delivered:
        return (
            f"在用户确认的范围「{scope_label}」中未找到可交付父块。"
            "请换关键词，或请用户改选知识库。"
        )

    # 切片 A：检索成功 → 按命中 category 写入候选 Policy Context（非归属）
    # 单库 scope 也用命中分类；__all__ 可能一次产生多个 ctx
    cats_for_policy = hit_categories
    if effective and effective not in cats_for_policy:
        cats_for_policy = [effective] + list(hit_categories)
    rag_id = begin_rag_call()
    created_ctx = append_policy_contexts_for_categories(
        cats_for_policy, rag_call_id=rag_id, query=query
    )
    ctx_note = ""
    if created_ctx:
        parts = [
            f"{c['ctx_id']}→{c['kb']}/{c['spec_id']}" for c in created_ctx
        ]
        ctx_note = f"；Policy Context: {', '.join(parts)}"

    return (
        f"检索成功（范围: {scope_label}；ParentDocument：子块定位 → 父块交付，同父已去重"
        f"{ctx_note}），"
        f"内容如下：\n" + "\n\n".join(delivered)
    )

