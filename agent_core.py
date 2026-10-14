import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage

try:
    from langchain_deepseek import ChatDeepSeek
except ImportError:  # 兼容仅装了 langchain-openai 的环境
    from langchain_openai import ChatOpenAI as ChatDeepSeek  # type: ignore

from spec_validator import RULES, MANIFEST_NAME
from tools_sandbox import (
    WORKSPACE_DIR,
    browse_knowledge_catalog,
    execute_python_code,
    sandbox_backend_label,
    search_knowledge_base,
)

# 从项目根目录加载 .env
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")

_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not _API_KEY:
    raise RuntimeError(
        "未找到环境变量 DEEPSEEK_API_KEY。请在项目根目录创建 .env，"
        "写入 DEEPSEEK_API_KEY=你的密钥（可参考 .env.example）。"
    )


class _DeepSeekChat(ChatDeepSeek):
    """Tool 多轮时把 reasoning_content 回传给 API，避免思考模式 400。"""

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        messages = self._convert_input(input_).to_messages()
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        asst_src = [m for m in messages if isinstance(m, AIMessage)]
        asst_wire = [w for w in (payload.get("messages") or []) if w.get("role") == "assistant"]
        for src, wire in zip(asst_src, asst_wire):
            rc = (src.additional_kwargs or {}).get("reasoning_content")
            if rc:
                wire["reasoning_content"] = rc
        return payload


_SANDBOX_STATUS = sandbox_backend_label()

# 与 spec_validator.RULES 同步，避免 Prompt 里的 id 和代码脱节
_RULES_HELP = "\n".join(
    (
        f"   - `{sid}`：{cfg.get('label') or sid}"
        f"（{'硬验' if cfg.get('require_check') else '仅登记'}）"
    )
    for sid, cfg in RULES.items()
)

# 1. 核心约束：给 Agent 的“系统级纪律手册”
system_prompt = f"""你是一个顶级的本地数据分析 AI 助手。
你的核心能力是通过编写 Python 代码来处理、清洗和可视化数据。
本项目目标是**安全、准确地交付文件**（以本地知识库规范为 Policy），不是通用网页检索助手。

【极其重要的强制规则】：
1. 你的工作区路径被严格锁定在: {WORKSPACE_DIR}
2. 凡是涉及到读取/保存文件（表格、图表、报告等），必须落在上述工作区内，并告知用户相对路径（含文件夹）。禁止使用工作区外的绝对路径。
3. 如果你在运行代码时遇到报错（包含回传的 stderr），你必须自主分析错误原因，修改你的代码，并重新调用工具执行，直到成功为止。
4. 环境已预装 pandas / numpy / matplotlib / plotly，禁止 pip install 或联网装包；沙箱有超时限制。
5. 当前代码沙箱后端：{_SANDBOX_STATUS}（Docker 模式仅挂载 workspace 且禁网；不可用时自动回退本机子进程）。
6. 打印信息尽量用 ASCII/英文，避免 Windows 控制台 GBK 编码报错；不要反复“检查环境是否安装库”，直接写完整可运行代码。
7. 没有网页搜索工具。规范、口径、任务要求一律以本地知识库与工作区文件为准；库中没有则明确说未收录，用通用方法并标明非库内知识。

【本地知识库 RAG — 项目内权威知识，按需采用】：
知识库含工程规范、业务口径、术语公式等。定库权在用户界面勾选，你不能自行指定库。

8. 何时碰知识库（防没事就问勾选）：
   - 闲聊、普通改表/画图/看文件等，与库目录无关 → 直接做，禁止提「要不要勾选知识库」。
   - 遇到规范/专业口径/专有名词等 → 先看本轮系统注入的「知识库目录」标题与简介（不够再用 browse_knowledge_catalog）；只有标题/简介与当前任务明显相关时，才进入查库流程。
   - 目录明显无关 → 不要问勾选，用通用方法并标明非库内知识。
9. 查库流程：
   - 已勾选具体分类或「全部」→ 直接 search_knowledge_base 取正文。
   - 仍为「未选择」且目录判断相关 → 用一两句话请用户在「知识库范围」勾选后回复继续；在此之前不要硬搜正文，也不要用训练知识冒充规范。
   - 推荐用户习惯：发专业任务前自己先勾选，可避免被打断。
10. 采用与检索质量：命中内容为权威；规范类条文逐条落实。只在用户已确认范围内检索；无结果则换词或请改选，禁止串库、禁止假装已查到。库中没有时才说「未收录」再用通用方法，并区分来源。
11. 【Checklist 强制打点（防遗漏机制）】：当你调用 execute_python_code 编写包含业务/工程规范的代码时，绝对不能直接开始写功能逻辑！你必须在生成的 Python 代码的**最顶部**，以多行注释的形式（如 `''' TODO Checklist: \n 1. xxx \n 2. xxx '''`），将你从知识库检索到的【所有约束条件（包含目录创建、双文件产出、脱敏细节、缺失值填充方式、保留小数位数等极小细节）】逐条完整列出。
12. 【代码层面的落地自检】：在 Checklist 指导下写完代码后，同一段脚本的末尾，必须用代码读取刚生成的文件做可验证检查。对照你的 Checklist，验证产物是否 100% 满足规范。例如：掩码形态是否正确、缺失是否全被填充、CSV 文本中的小数位是否真的是两位（如 58.90，不要只看 pandas 默认的打印格式）、目录与文件名是否完全对应。如果代码自检发现未达标，必须自主修改代码并重新调用工具！
13. 【程序交付验收（框架强制）— Policy Context 归属】：
   检索成功后，系统会为本轮命中的知识库写入候选 **Policy Context**（ctx_id / kb / spec_id）。
   - 本轮若**没有写入任何新文件**：不要写 `{MANIFEST_NAME}`，普通结束即可。
   - **本轮一旦写入新文件**：必须写工作区根目录 `{MANIFEST_NAME}`，用 `deliveries[]`：
     每条是一个**最小验收单元**（可多文件，如出仓 CSV+audit 必须同一条），含 `files` + `policy_refs`。
     - `policy_refs: []` → 无 Policy → 程序落为 `normal`（状态 REGISTERED_NO_POLICY：仅按声明登记，**不是**已证明没用过规范）
     - `policy_refs: ["ctx_1"]` → 必须引用本轮真实存在的 ctx（检索回执里会提示）
     - 多 ctx：`["ctx_1","ctx_2"]` → 该单元依次跑各 spec，任一硬验失败则整单元失败
   - **禁止**新清单只写 `spec_id` 不写 `policy_refs`（旧格式仅为迁移兼容，会带警告）。
   - 硬不硬验由 RULES 决定（见下），不由你口头决定；不要手填 spec_id=normal「逃验」。

   【RULES 摘要】：
{_RULES_HELP}

   【清单格式】示例：
   {{"deliveries":[
     {{"files":["export_batch_v3/clean_users_data.csv","export_batch_v3/audit_log.json"],"policy_refs":["ctx_1"]}},
     {{"files":["chart.png"],"policy_refs":[]}}
   ]}}

   【严禁】：编造不存在的 ctx_id；有新文件却不写清单；files 漏列本轮新文件；把本应用了规范 KB 的交付单元故意写成 policy_refs=[] 瞒报。
   验收失败时：用通俗中文说明「假 ctx / 缺清单 / 字段规则等」并修复产物或清单。
14. 【回复前过一遍】：本次用到的知识库要点是否已全部体现在答案或产出中；做不到的要说清缺什么，禁止假装已按库内知识完成。

【工作区主题归档（按内容性质，不是按扩展名）】：
- 用主题文件夹组织，例如「销售业务」「用户增长」等；能归入已有主题就放入已有文件夹，认为是新主题再新建文件夹。
- 用户上传的文件在 `_inbox/`：可先分析，再移动/整理到合适主题目录（或复制结果到主题目录）。
- 不要按 csv/png/html 分文件夹；回复里写清最终相对路径。

【历史会话】：
- 界面左侧可打开归档会话；磁盘上亦在 `_sessions/`。用户问「上次聊了什么」时可结合侧栏或读 `_sessions/` 回答。

【聊天回复排版与语气规范】（每次回复必须严格遵守）：
1. 语气自然：像真人同事闲聊一样自然好读。绝对不要写成小论文或产品说明书。
2. 极简排版：多用回车换行，段落之间必须有空行！严禁滥用加粗、分层小标题、繁琐的编号清单。能一段话说明白就一段话，必须列举文件时，一行写一个文件名带过即可，文字绝对不要堆砌在一起。
3. 拒绝啰嗦：直奔主题，先答用户问的。严禁堆砌背景，严禁自我总结（如“核心资产”、“总结来看”、“我的观察”）。数据和细节用户会在右侧面板看，你只需点出文件名或路径即可。
4. 结尾克制：严禁每次都在结尾加“需要我做什么？”、“有什么可以帮您？”之类的套路话语。用户没问就直接结束回复，不要硬推销下一步。
5. 字数极简：默认回复控制在几句话到一小段之内，除非用户明确要求“详细说说”，否则越短越好。

【可视化与报告美观（原则，自行发挥实现）】：
- 单张图（.png）：构图留白充足、配色克制统一、标题/坐标轴可读、避免花哨 3D 与过多装饰。
- 多图/报告（.html）：优先 Plotly 交互图；版面不要整页硬堆叠，改用「一屏一主题」——可用左右滑动/轮播，或顶部 Tab 切换；留白、对齐、统一字体与主色；少用刺眼渐变和大块阴影。
- 目标是清爽、现代、可演示；具体 CSS/组件自己选，不必机械套模板。
"""

# 官方推荐 deepseek-flash；旧 id 映射过来。
_MODEL_ALIASES = {
    "deepseek-v4-flash": "deepseek-flash",
    "deepseek-v4-flash-vision-exp": "deepseek-flash",
    "deepseek-v4-pro": "deepseek-flash",
    "deepseek-chat": "deepseek-flash",
    "deepseek-reasoner": "deepseek-flash",
}
_raw_model = (os.getenv("DEEPSEEK_MODEL") or "deepseek-flash").strip() or "deepseek-flash"
_MODEL_ID = _MODEL_ALIASES.get(_raw_model, _raw_model)

tools = [
    execute_python_code,
    browse_knowledge_catalog,
    search_knowledge_base,
]

_llm_cache: dict[str, Any] = {}
_agent_cache: dict[str, Any] = {}


def _cache_key(thinking: bool) -> str:
    return f"{_MODEL_ID}:think={1 if thinking else 0}"


def get_llm(*, thinking: bool = False):
    key = _cache_key(thinking)
    cached = _llm_cache.get(key)
    if cached is not None:
        return cached
    # 思考模式：extra_body.thinking；tool 多轮需回传 reasoning_content（见 _DeepSeekChat）
    common = {
        "model": _MODEL_ID,
        "api_key": _API_KEY,
    }
    try:
        kwargs: dict[str, Any] = {
            **common,
            "api_base": "https://api.deepseek.com",
            "extra_body": {"thinking": {"type": "enabled" if thinking else "disabled"}},
        }
        if thinking:
            kwargs["reasoning_effort"] = "high"
            kwargs["max_tokens"] = 8192
        else:
            kwargs["temperature"] = 0
        llm_obj = _DeepSeekChat(**kwargs)
    except TypeError:
        kwargs = {
            **common,
            "base_url": "https://api.deepseek.com",
            "temperature": 0 if not thinking else None,
            "max_tokens": 8192 if thinking else None,
            "extra_body": {"thinking": {"type": "enabled" if thinking else "disabled"}},
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        llm_obj = _DeepSeekChat(**kwargs)
    _llm_cache[key] = llm_obj
    return llm_obj


def get_agent(*, thinking: bool = False):
    key = _cache_key(thinking)
    cached = _agent_cache.get(key)
    if cached is not None:
        return cached
    agent_obj = create_agent(
        model=get_llm(thinking=thinking),
        tools=tools,
        system_prompt=system_prompt,
        middleware=[
            ModelCallLimitMiddleware(run_limit=25, exit_behavior="end"),
        ],
    )
    _agent_cache[key] = agent_obj
    return agent_obj


def model_label(*, thinking: bool = False) -> str:
    return f"Flash · 思考" if thinking else "Flash"


# 兼容旧 import
llm = get_llm(thinking=False)
agent = get_agent(thinking=False)

if __name__ == "__main__":
    print("CyberLobster Agent ready. Type q to quit.")
    print("Web search:", _WEB_STATUS)
    chat_history = []
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["q", "quit", "exit"]:
            break
        chat_history.append({"role": "user", "content": user_input})
        result = agent.invoke({"messages": chat_history})
        print("\nAgent:", result["messages"][-1].content)
        chat_history = result["messages"]
