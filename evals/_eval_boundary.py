"""
项目一 · E3 沙箱边界评测（Minimal Evaluation）

测什么：
  - 代码契约：Docker 路径是否声明 --network none、仅挂载 workspace
  - 行为矩阵：workspace 内写 / 相对越界 / 绝对路径越界 / 出网
    · local：实测本机子进程（cwd=workspace，与 tools_sandbox 一致）
    · docker：有 CLI+镜像则实测；否则 SKIP，并写明设计期望

不测什么：
  - 「更安全」笼统结论、逃逸 exploit、端到端 Agent

用法：
  cd cyber_lobster_project
  python evals/_eval_boundary.py

报告：evals/EVAL_REPORT_BOUNDARY.md
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

from _bootstrap import EVAL_DIR, ROOT

WORKSPACE = ROOT / "workspace"
SANDBOX_SRC = ROOT / "tools_sandbox.py"
IMAGE = (os.getenv("SANDBOX_DOCKER_IMAGE") or "cyberlobster-sandbox:py311").strip()
ESCAPE_MARKER = ROOT / "_eval_boundary_escape.tmp"
WS_OK = WORKSPACE / "_eval_boundary_ok.txt"
OUTSIDE_DIR: Path | None = None


def _run_local(code: str) -> tuple[int, str, str]:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    script = WORKSPACE / "_eval_boundary_script.py"
    script.write_text(code, encoding="utf-8")
    try:
        r = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(WORKSPACE),
            capture_output=True,
            text=True,
            timeout=15,
        )
        return r.returncode, r.stdout or "", r.stderr or ""
    finally:
        script.unlink(missing_ok=True)


def _docker_cli() -> str | None:
    return shutil.which("docker")


def _docker_info_ok() -> bool:
    cli = _docker_cli()
    if not cli:
        return False
    try:
        r = subprocess.run([cli, "info"], capture_output=True, text=True, timeout=8)
        return r.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _image_present() -> bool:
    cli = _docker_cli()
    if not cli:
        return False
    try:
        r = subprocess.run(
            [cli, "image", "inspect", IMAGE],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return r.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _run_docker(code: str) -> tuple[str, int | None, str, str]:
    """返回 (status, returncode, stdout, stderr). status=ok|skip|error"""
    cli = _docker_cli()
    if not cli or not _docker_info_ok():
        return "skip", None, "", "docker CLI/daemon 不可用"
    if not _image_present():
        return "skip", None, "", f"镜像缺失: {IMAGE}"

    WORKSPACE.mkdir(parents=True, exist_ok=True)
    script = WORKSPACE / "_eval_boundary_script.py"
    script.write_text(code, encoding="utf-8")
    host = str(WORKSPACE.resolve())
    mount = f"{host}:/workspace"
    cid = None
    try:
        cr = subprocess.run(
            [
                cli,
                "create",
                "--network",
                "none",
                "-w",
                "/workspace",
                "-v",
                mount,
                IMAGE,
                "python",
                "/workspace/_eval_boundary_script.py",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if cr.returncode != 0:
            return "error", cr.returncode, cr.stdout or "", cr.stderr or ""
        cid = (cr.stdout or "").strip()
        start = subprocess.run(
            [cli, "start", "-a", cid],
            capture_output=True,
            text=True,
            timeout=20,
        )
        return "ok", start.returncode, start.stdout or "", start.stderr or ""
    except subprocess.TimeoutExpired:
        return "error", None, "", "timeout"
    finally:
        if cid and cli:
            try:
                subprocess.run([cli, "rm", "-f", cid], capture_output=True, timeout=8)
            except (OSError, subprocess.TimeoutExpired):
                pass
        script.unlink(missing_ok=True)


def eval_docker_contract() -> dict:
    """静态：源码是否按设计声明禁网 + 仅挂载 workspace。"""
    src = SANDBOX_SRC.read_text(encoding="utf-8")
    has_network_none = bool(
        re.search(r'["\']--network["\']\s*,\s*["\']none["\']', src)
        or re.search(r"--network[\s\n]+none", src)
    )
    has_mount_helper = "_workspace_mount_spec" in src and "-v" in src
    # 创建命令段不应出现额外 -v（粗检：_execute_in_docker 内 -v 次数）
    m = re.search(r"def _execute_in_docker\([\s\S]*?\n(?=def |\Z)", src)
    block = m.group(0) if m else ""
    v_flags = len(re.findall(r'["\']-v["\']', block))
    # 允许 1 次 -v（workspace）；禁止把整个项目根再挂进去的明显写法
    no_root_mount = "/workspace" in block and "knowledge_docs" not in block

    ok = has_network_none and has_mount_helper and v_flags == 1 and no_root_mount
    return {
        "name": "Docker代码契约",
        "network_none": has_network_none,
        "single_workspace_mount": v_flags == 1 and has_mount_helper,
        "v_flag_count_in_execute": v_flags,
        "no_extra_root_mount_hint": no_root_mount,
        "ok": ok,
    }


def _cleanup_artifacts() -> None:
    for p in (WS_OK, ESCAPE_MARKER):
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass
    global OUTSIDE_DIR
    if OUTSIDE_DIR and OUTSIDE_DIR.is_dir():
        shutil.rmtree(OUTSIDE_DIR, ignore_errors=True)
        OUTSIDE_DIR = None


def eval_behavior_matrix() -> dict:
    """
    行为用例。期望（诚实）：
      local  — 进程级软隔离：workspace 内写 ✓；越界/出网通常仍可能成功
      docker — 设计期望：仅挂载可见；越界相对路径落到宿主机未挂载区应失败或不可见；出网失败
    """
    global OUTSIDE_DIR
    _cleanup_artifacts()
    OUTSIDE_DIR = Path(tempfile.mkdtemp(prefix="cl_boundary_abs_"))
    abs_target = OUTSIDE_DIR / "abs_write.txt"

    cases = []

    # 1) workspace 内写
    code_ok = textwrap.dedent(
        f"""
        from pathlib import Path
        Path({WS_OK.name!r}).write_text("ok", encoding="utf-8")
        print("WROTE_OK")
        """
    )
    rc, out, err = _run_local(code_ok)
    local_ok = rc == 0 and WS_OK.is_file() and "WROTE_OK" in out
    WS_OK.unlink(missing_ok=True)

    d_status, d_rc, d_out, d_err = _run_docker(code_ok)
    if d_status == "ok":
        docker_ok = d_rc == 0 and WS_OK.is_file()
        WS_OK.unlink(missing_ok=True)
    else:
        docker_ok = None

    cases.append(
        {
            "id": "ws_write",
            "desc": "写入 workspace 内相对路径",
            "expect_local": "allow",
            "expect_docker": "allow",
            "local": {"allowed": local_ok, "rc": rc, "note": (err[:120] if err else out[:80])},
            "docker": {
                "status": d_status,
                "allowed": docker_ok,
                "rc": d_rc,
                "note": d_err[:200] if d_status != "ok" else (d_out[:80] if d_out else ""),
            },
        }
    )

    # 2) 相对越界 ../（cwd=workspace → 落到项目根）
    code_esc = textwrap.dedent(
        f"""
        from pathlib import Path
        Path("..").joinpath({ESCAPE_MARKER.name!r}).write_text("escaped", encoding="utf-8")
        print("WROTE_ESCAPE")
        """
    )
    ESCAPE_MARKER.unlink(missing_ok=True)
    rc, out, err = _run_local(code_esc)
    local_esc = ESCAPE_MARKER.is_file()
    ESCAPE_MARKER.unlink(missing_ok=True)

    d_status, d_rc, d_out, d_err = _run_docker(code_esc)
    if d_status == "ok":
        # 容器内 ../ 若未挂载项目根，文件不应出现在宿主机 ESCAPE_MARKER
        docker_esc = ESCAPE_MARKER.is_file()
        ESCAPE_MARKER.unlink(missing_ok=True)
        # allowed=True means escape succeeded (bad for isolation)
    else:
        docker_esc = None

    cases.append(
        {
            "id": "rel_escape",
            "desc": "相对路径 ../ 写到 workspace 外",
            "expect_local": "allow（软隔离，预期可逃出）",
            "expect_docker": "block（宿主机项目根未挂载，文件不应出现）",
            "local": {"allowed": local_esc, "rc": rc, "note": "文件出现在项目根=逃出"},
            "docker": {
                "status": d_status,
                "allowed": docker_esc,  # True=逃出成功（坏）
                "rc": d_rc,
                "note": (
                    d_err[:200]
                    if d_status != "ok"
                    else ("宿主机见逃逸文件" if docker_esc else "宿主机未见逃逸文件")
                ),
            },
        }
    )

    # 3) 绝对路径写到临时目录（workspace 外）
    code_abs = textwrap.dedent(
        f"""
        from pathlib import Path
        Path({str(abs_target)!r}).write_text("abs", encoding="utf-8")
        print("WROTE_ABS")
        """
    )
    if abs_target.exists():
        abs_target.unlink()
    rc, out, err = _run_local(code_abs)
    local_abs = abs_target.is_file()
    if abs_target.exists():
        abs_target.unlink()

    d_status, d_rc, d_out, d_err = _run_docker(code_abs)
    if d_status == "ok":
        docker_abs = abs_target.is_file()
        if abs_target.exists():
            abs_target.unlink()
    else:
        docker_abs = None

    cases.append(
        {
            "id": "abs_escape",
            "desc": f"绝对路径写入临时目录（{OUTSIDE_DIR}）",
            "expect_local": "allow",
            "expect_docker": "block（路径在容器内无意义或不可写到宿主机该处）",
            "local": {"allowed": local_abs, "rc": rc, "note": ""},
            "docker": {
                "status": d_status,
                "allowed": docker_abs,
                "rc": d_rc,
                "note": (
                    d_err[:200]
                    if d_status != "ok"
                    else ("宿主机见文件" if docker_abs else "宿主机未见文件")
                ),
            },
        }
    )

    # 4) 出网
    code_net = textwrap.dedent(
        """
        import urllib.request
        try:
            urllib.request.urlopen("https://example.com", timeout=5)
            print("NET_OK")
        except Exception as e:
            print("NET_FAIL:" + type(e).__name__)
        """
    )
    rc, out, err = _run_local(code_net)
    local_net = "NET_OK" in out

    d_status, d_rc, d_out, d_err = _run_docker(code_net)
    if d_status == "ok":
        docker_net = "NET_OK" in (d_out or "")
    else:
        docker_net = None

    cases.append(
        {
            "id": "network",
            "desc": "urllib 访问 https://example.com",
            "expect_local": "allow（本机有网则可）",
            "expect_docker": "block（--network none）",
            "local": {"allowed": local_net, "rc": rc, "note": out.strip()[:80]},
            "docker": {
                "status": d_status,
                "allowed": docker_net,
                "rc": d_rc,
                "note": (
                    d_err[:200]
                    if d_status != "ok"
                    else ((d_out or "") + (d_err or ""))[:120]
                ),
            },
        }
    )

    # 判定：local 侧「软隔离」故事成立 = ws allow + escape allow + abs allow
    # docker 侧若跑通：ws allow + escape not + abs not + net not
    local_story_ok = (
        cases[0]["local"]["allowed"] is True
        and cases[1]["local"]["allowed"] is True
        and cases[2]["local"]["allowed"] is True
    )
    docker_ran = all(c["docker"]["status"] == "ok" for c in cases)
    docker_story_ok = None
    if docker_ran:
        docker_story_ok = (
            cases[0]["docker"]["allowed"] is True
            and cases[1]["docker"]["allowed"] is False
            and cases[2]["docker"]["allowed"] is False
            and cases[3]["docker"]["allowed"] is False
        )

    return {
        "name": "边界行为矩阵",
        "cases": cases,
        "local_soft_isolation_confirmed": local_story_ok,
        "docker_hard_isolation_confirmed": docker_story_ok,
        "docker_ran": docker_ran,
        "ok": local_story_ok and (docker_story_ok is not False),
        # ok=True if local story holds and docker either skipped or passed; False only if docker ran and failed
    }


def _cell(allowed, status=None) -> str:
    if status and status != "ok":
        return f"SKIP({status})"
    if allowed is True:
        return "允许"
    if allowed is False:
        return "阻断/未落到宿主机"
    return "—"


def write_report(contract: dict, matrix: dict) -> Path:
    docker_avail = _docker_info_ok()
    image_ok = _image_present() if docker_avail else False
    lines = [
        "# 本地数据分析 Agent · 沙箱边界评测（E3）",
        "",
        "> 脚本：`_eval_boundary.py`（不调用 LLM）  ",
        "> 对照：`tools_sandbox._execute_in_docker` / `_execute_local_subprocess`",
        "",
        "## 环境",
        "",
        f"| 项 | 值 |",
        f"|----|----|",
        f"| Docker CLI/daemon | {'可用' if docker_avail else '不可用'} |",
        f"| 镜像 `{IMAGE}` | {'存在' if image_ok else '不存在/未测'} |",
        f"| 本机实测后端 | **local**（与当前 resolve 一致时） |",
        "",
        "## 1. Docker 代码契约（静态）",
        "",
        f"| 检查项 | 结果 |",
        f"|--------|------|",
        f"| `--network none` | {'✓' if contract['network_none'] else '✗'} |",
        f"| 仅一次 `-v` workspace 挂载 | {'✓' if contract['single_workspace_mount'] else '✗'}（v_count={contract['v_flag_count_in_execute']}） |",
        f"| 未把 knowledge 等根目录挂进容器 | {'✓' if contract['no_extra_root_mount_hint'] else '✗'} |",
        f"| 契约总评 | **{'PASS' if contract['ok'] else 'FAIL'}** |",
        "",
        "## 2. 行为矩阵",
        "",
        "| 用例 | 设计期望 local | 实测 local | 设计期望 docker | 实测 docker |",
        "|------|----------------|------------|-----------------|-------------|",
    ]
    for c in matrix["cases"]:
        loc = _cell(c["local"]["allowed"])
        dock = _cell(c["docker"]["allowed"], c["docker"]["status"])
        lines.append(
            f"| {c['desc']} | {c['expect_local']} | {loc} | {c['expect_docker']} | {dock} |"
        )

    lines.extend(
        [
            "",
            "## 3. 结论（可写进 Evaluation / 口述）",
            "",
            f"- **Local 软隔离已确认**：{matrix['local_soft_isolation_confirmed']} "
            "（workspace 内可写；`../` 与绝对路径可写到 workspace 外；出网视本机网络而定）。",
            f"- **Docker 硬隔离**：",
        ]
    )
    if matrix["docker_ran"]:
        lines.append(
            f"  实测确认={matrix['docker_hard_isolation_confirmed']} "
            "（期望：内写允许；相对/绝对越界不落到宿主机目标；出网失败）。"
        )
    else:
        lines.append(
            "  **本机未跑通容器**（无 Docker 或无镜像）→ 行为矩阵 docker 列为 SKIP；"
            "硬隔离结论目前仅依赖**代码契约**（`--network none` + 单 workspace 挂载），"
            "**不能**写成「已在本机用容器实测阻断」。"
        )

    lines.extend(
        [
            "",
            "### 诚实表述模板",
            "",
            (
                "> Docker 模式下代码路径声明了 `--network none` 且仅挂载 workspace；"
                "无 Docker 时回退本机子进程。本机边界实测表明 local 为进程级软隔离"
                "（越界写与出网在条件允许时可成功）。容器硬隔离"
                + (
                    "已在本机用同一组用例实测通过。"
                    if matrix.get("docker_hard_isolation_confirmed")
                    else "需在具备 Docker 与沙箱镜像的环境复跑本脚本后方可声称实测通过。"
                )
            ),
            "",
            "## 4. 局限",
            "",
            "- 不是安全攻防评测；未覆盖符号链接、Docker socket、内核逃逸等。",
            "- local 用与生产相同的 `cwd=workspace` 子进程模型，但未强制走 `execute_python_code` 的 COW preamble。",
            "- docker 实测复用与源码相同的 create 参数；若日后改挂载策略需重跑。",
            "",
        ]
    )

    out = EVAL_DIR / "EVAL_REPORT_BOUNDARY.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    print("=== Docker contract ===")
    contract = eval_docker_contract()
    print(contract)

    print("\n=== Behavior matrix ===")
    try:
        matrix = eval_behavior_matrix()
        for c in matrix["cases"]:
            print(c["id"], "local=", c["local"], "docker=", c["docker"])
        print(
            "local_soft=",
            matrix["local_soft_isolation_confirmed"],
            "docker_hard=",
            matrix["docker_hard_isolation_confirmed"],
            "docker_ran=",
            matrix["docker_ran"],
        )
    finally:
        _cleanup_artifacts()

    path = write_report(contract, matrix)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
