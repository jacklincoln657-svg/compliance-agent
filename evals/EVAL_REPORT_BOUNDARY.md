# 本地数据分析 Agent · 沙箱边界评测（E3）

> 脚本：`_eval_boundary.py`（不调用 LLM）  
> 对照：`tools_sandbox._execute_in_docker` / `_execute_local_subprocess`

## 环境

| 项 | 值 |
|----|----|
| Docker CLI/daemon | 不可用 |
| 镜像 `cyberlobster-sandbox:py311` | 不存在/未测 |
| 本机实测后端 | **local**（与当前 resolve 一致时） |

## 1. Docker 代码契约（静态）

| 检查项 | 结果 |
|--------|------|
| `--network none` | ✓ |
| 仅一次 `-v` workspace 挂载 | ✓（v_count=1） |
| 未把 knowledge 等根目录挂进容器 | ✓ |
| 契约总评 | **PASS** |

## 2. 行为矩阵

| 用例 | 设计期望 local | 实测 local | 设计期望 docker | 实测 docker |
|------|----------------|------------|-----------------|-------------|
| 写入 workspace 内相对路径 | allow | 允许 | allow | SKIP(skip) |
| 相对路径 ../ 写到 workspace 外 | allow（软隔离，预期可逃出） | 允许 | block（宿主机项目根未挂载，文件不应出现） | SKIP(skip) |
| 绝对路径写入临时目录（C:\Users\FAC\AppData\Local\Temp\cl_boundary_abs_eslt78zp） | allow | 允许 | block（路径在容器内无意义或不可写到宿主机该处） | SKIP(skip) |
| urllib 访问 https://example.com | allow（本机有网则可） | 允许 | block（--network none） | SKIP(skip) |

## 3. 结论（可写进 Evaluation / 口述）

- **Local 软隔离已确认**：True （workspace 内可写；`../` 与绝对路径可写到 workspace 外；出网视本机网络而定）。
- **Docker 硬隔离**：
  **本机未跑通容器**（无 Docker 或无镜像）→ 行为矩阵 docker 列为 SKIP；硬隔离结论目前仅依赖**代码契约**（`--network none` + 单 workspace 挂载），**不能**写成「已在本机用容器实测阻断」。

### 诚实表述模板

> Docker 模式下代码路径声明了 `--network none` 且仅挂载 workspace；无 Docker 时回退本机子进程。本机边界实测表明 local 为进程级软隔离（越界写与出网在条件允许时可成功）。容器硬隔离需在具备 Docker 与沙箱镜像的环境复跑本脚本后方可声称实测通过。

## 4. 局限

- 不是安全攻防评测；未覆盖符号链接、Docker socket、内核逃逸等。
- local 用与生产相同的 `cwd=workspace` 子进程模型，但未强制走 `execute_python_code` 的 COW preamble。
- docker 实测复用与源码相同的 create 参数；若日后改挂载策略需重跑。
