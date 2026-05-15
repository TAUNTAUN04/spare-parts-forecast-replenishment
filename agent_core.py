"""
=================================================================
 Agent 核心：智谱 AI GLM tool-use 循环（OpenAI 兼容协议）
-----------------------------------------------------------------
 模式：手动 agentic loop
   每轮：
     1) client.chat.completions.create(messages, tools, tool_choice="auto")
     2) 检查 choices[0].finish_reason：
          - "stop" / "length" / "sensitive"  → 结束
          - "tool_calls"                      → 执行所有 tool_calls，把结果拼成 role:"tool" 消息
     3) 把 assistant 整条消息（含 tool_calls 数组）追加到 messages
     4) 每个工具结果作为一条 {role:"tool", tool_call_id, content} 追加，回到 1

 关键 API 约束（与 Anthropic 不同的地方，注意）：
   · system 不是单独参数，是 messages[0] 的 {role:"system"} 消息
   · tool 定义要包一层 {type:"function", function:{...}}（见 agent_tools.TOOL_SCHEMAS_OPENAI）
   · assistant 回复含工具调用时，tool_calls 是 message.tool_calls 数组（不是 content 块）
   · function.arguments 是 **JSON 字符串**，不是 dict，要 json.loads
   · tool 返回必须用 {role:"tool", tool_call_id:..., content: 字符串}
     （tool_call_id 必须严格匹配上一轮 message.tool_calls[i].id）
=================================================================
"""
from __future__ import annotations

import json
import os
import sys
from typing import Generator

# Windows 上 Streamlit 子进程的 stdout/stderr 默认编码可能不是 UTF-8，
# 当 print 中文或子库写日志时会触发 'ascii' codec encode 错误。强制 UTF-8。
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from zhipuai import ZhipuAI

from agent_tools import TOOL_SCHEMAS_OPENAI, dispatch_tool

# GLM-4 系列可选：glm-4-plus（旗舰，工具调用稳定，推荐）/ glm-4-air（快）/ glm-4-flash（免费/便宜）
# 模型清单与价格见 https://open.bigmodel.cn/pricing
MODEL = "glm-4-plus"
MAX_TOKENS = 4096
MAX_ITERS = 8   # 防止 LLM 工具调用陷入死循环


SYSTEM_PROMPT = """你是「智能备件需求预测与补货决策助手」，服务于一位**没有供应链科班背景**的研究生用户（正在系统学习供应链领域知识）。

# 你的能力
你可以调用 5 个工具操作真实数据：
1. `list_skus`              — 列出可用 SKU 和仓库
2. `get_inventory_status`   — 查某 SKU 当前库存（在手 / 在途 / 安全库存 / 提前期）
3. `forecast_demand`        — Prophet 预测未来 N 天需求并返回 MAE
4. `compute_replenishment`  — 自动算 ROP（再订货点）+ 订到点 S + 建议下单量
5. `place_order`            — **沙盒**模拟下单（仅写日志，不会真采购）

# 工作流程
- 用户提问 → 先决定是否需要调用工具
- 简单问题可一次性调用 `compute_replenishment`（它内部会自动跑预测 + 拉库存）
- **下单前必须先告知用户建议数量并请其确认**，得到肯定答复后再调 `place_order`
- 不要凭空编造数字 — 任何具体指标（库存、MAE、ROP、补货量）都要来自工具

# 回复风格
- **简洁、决策导向**；先给结论再给依据
- 涉及关键概念时**顺带科普一句**（如"ROP 就是再订货点：库存跌破这条线就该下单"），用户在学习
- 用**中文**，关键术语后括号附英文（如 ROP / Safety Stock）便于学习与查文献
- 适度用 markdown 表格列指标，便于阅读
- 不要在没调用工具的情况下杜撰数字

# 边界
- 你**只是辅助决策**，下单是沙盒模拟，不会真正影响业务
- 数据范围：20 个 SKU（SKU001~SKU020）× 2 个仓库（WH_A / WH_B），日期约 2025-05-11 到 2026-01-15
- 用户不确定 SKU/仓库时，主动调 `list_skus` 给出选择
"""


def get_client(api_key: str | None = None) -> ZhipuAI:
    """构造 ZhipuAI 客户端。优先用传入的 key，否则用环境变量 ZHIPUAI_API_KEY。"""
    key = api_key or os.environ.get("ZHIPUAI_API_KEY")
    if not key:
        raise RuntimeError(
            "未提供智谱 AI API Key。请在侧边栏输入或设置环境变量 ZHIPUAI_API_KEY。"
            "Key 申请：https://open.bigmodel.cn/usercenter/apikeys"
        )
    return ZhipuAI(api_key=key)


def make_initial_messages() -> list[dict]:
    """生成新对话的初始 messages（system 作为第一条）。"""
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def run_agent_turn(
    client: ZhipuAI,
    messages: list[dict],
    max_iters: int = MAX_ITERS,
) -> Generator[tuple[str, dict | str], None, list[dict]]:
    """运行一次完整的 agent 回合（用户已经把新消息追加进 messages）。

    messages 格式（OpenAI 风格）：
      [
        {role: "system",    content: ...},
        {role: "user",      content: "你好"},
        {role: "assistant", content: "...", tool_calls: [...]},   # 模型回复（含工具调用）
        {role: "tool",      tool_call_id: "call_xxx", content: "..."},  # 工具返回
        ...
      ]

    Yields (event_type, payload)：
      - ("text",        str)
      - ("tool_use",    {id, name, input})
      - ("tool_result", {id, name, result_preview, is_error})
      - ("done",        {stop_reason, usage_dict?})

    返回值是更新后的 messages 列表。
    """
    for _ in range(max_iters):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS_OPENAI,
            tool_choice="auto",
            max_tokens=MAX_TOKENS,
        )

        choice = response.choices[0]
        msg = choice.message

        # ===== 关键：把完整的 assistant 消息（含 tool_calls）追加到 messages =====
        # 哪怕只有 tool_calls 没有 content，也要保留这一条 — 后续 turn 才能正确关联 tool_call_id
        assistant_entry: dict = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,   # 注意：JSON 字符串
                    },
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_entry)

        # 把文本流给 UI
        if msg.content and msg.content.strip():
            yield ("text", msg.content)

        # 把工具调用流给 UI
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    parsed_input = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError:
                    parsed_input = {"_raw_arguments": tc.function.arguments}
                yield ("tool_use",
                       {"id": tc.id, "name": tc.function.name, "input": parsed_input})

        finish = choice.finish_reason  # "stop" | "tool_calls" | "length" | "sensitive" | ...

        if finish in ("stop", "length"):
            yield ("done", {
                "stop_reason": finish,
                "usage": _usage_dict(response.usage),
            })
            return messages

        if finish == "sensitive":
            yield ("done", {"stop_reason": "sensitive（智谱内容安全策略拦截）"})
            return messages

        if finish == "tool_calls":
            # 逐个执行工具，把结果作为 role:"tool" 消息塞回
            for tc in (msg.tool_calls or []):
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    result = dispatch_tool(tc.function.name, args)
                    is_error = False
                    result_text = json.dumps(result, ensure_ascii=False)
                except Exception as e:
                    is_error = True
                    result_text = f"工具执行失败：{type(e).__name__}: {e}"

                yield ("tool_result", {
                    "id": tc.id,
                    "name": tc.function.name,
                    "result_preview": result_text[:1500] +
                                      ("..." if len(result_text) > 1500 else ""),
                    "is_error": is_error,
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_text,
                })
            continue  # 进入下一轮

        # 未知 finish_reason，安全退出
        yield ("done", {"stop_reason": finish, "usage": _usage_dict(response.usage)})
        return messages

    yield ("done", {"stop_reason": "max_iters_exceeded"})
    return messages


def _usage_dict(usage) -> dict:
    """把 ZhipuAI 的 usage 对象转成可序列化 dict。
    GLM 的字段是 OpenAI 风格：prompt_tokens / completion_tokens / total_tokens。
    （GLM 也有 prompt 缓存，但暂未在 usage 中独立暴露；后续若升级 SDK 再补字段）
    """
    return {
        "prompt_tokens":     getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "total_tokens":      getattr(usage, "total_tokens", 0) or 0,
    }


# ---------- 命令行自检 ----------
if __name__ == "__main__":
    """直接 python agent_core.py "你的问题" 跑一次。需要 ZHIPUAI_API_KEY 环境变量。"""
    import sys
    user_msg = sys.argv[1] if len(sys.argv) > 1 else "SKU005 在 WH_A 仓库需要补货吗？要补多少？"
    client = get_client()
    msgs: list[dict] = make_initial_messages()
    msgs.append({"role": "user", "content": user_msg})
    print(f"\n>>> 用户：{user_msg}\n")
    for ev_type, payload in run_agent_turn(client, msgs):
        if ev_type == "text":
            print(payload, end="", flush=True)
        elif ev_type == "tool_use":
            print(f"\n  [调用工具] {payload['name']}({json.dumps(payload['input'], ensure_ascii=False)})")
        elif ev_type == "tool_result":
            tag = "❌" if payload["is_error"] else "✓"
            preview = payload["result_preview"][:200]
            print(f"  [{tag} {payload['name']} 返回] {preview}{'...' if len(payload['result_preview']) > 200 else ''}")
        elif ev_type == "done":
            print(f"\n\n--- 结束（finish_reason={payload['stop_reason']}）---")
            if payload.get("usage"):
                u = payload["usage"]
                print(f"    tokens: prompt={u['prompt_tokens']}  "
                      f"completion={u['completion_tokens']}  "
                      f"total={u['total_tokens']}")
