"""
============================================================================
 Streamlit 看板：备件需求预测 + 补货建议  +  AI 对话助手
----------------------------------------------------------------------------
 启动：streamlit run 02_streamlit_app.py
 两个标签页：
   1. 主看板      — 选 SKU/仓库 → Prophet 预测曲线 + 补货建议
   2. AI 助手     — 自然语言提问，GLM Agent 调用工具回答 + 沙盒下单
============================================================================
"""

import json
import logging
import os
import sys
import traceback
import warnings

# 强制 UTF-8（防御 Windows 子进程 stdout/stderr 编码非 UTF-8 引发的中文报错）
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

# ---- 中文字体：Windows 上用微软雅黑/黑体，否则字形会渲染成方框 □ ----
# rcParams 是 matplotlib 模块级配置，必须在创建 Figure 之前设。
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",     # Windows 7+ 默认
    "SimHei",              # Windows 备用
    "PingFang SC",         # macOS
    "Heiti SC",            # macOS 备用
    "Noto Sans CJK SC",    # Linux
    "Arial Unicode MS",    # 通用 CJK 兜底
]
plt.rcParams["axes.unicode_minus"] = False   # 负号也要正常显示

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")
from prophet import Prophet  # noqa: E402

# Agent 相关 — 仅在用户切到 AI 标签页时才会真正用到
from agent_core import (  # noqa: E402
    get_client, run_agent_turn, make_initial_messages,
    MODEL as AGENT_MODEL,
)

DATA_PATH = "inventory_replenishment_timeseries_10000.csv"

st.set_page_config(page_title="备件预测 + AI 助手", layout="wide")
st.title("智能备件需求预测与补货决策")
st.caption(f"Prophet × 动态 ROP × Streamlit  +  GLM Agent（{AGENT_MODEL}）")


# ---------- 公共：数据 + Prophet 缓存 ----------
@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv(DATA_PATH, parse_dates=["date"])


@st.cache_resource(show_spinner=True)
def fit_prophet(_train_df: pd.DataFrame, _cache_key: str):
    """缓存 key 显式传入，避免 streamlit 用 DataFrame 哈希出错。"""
    m = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.95,
    )
    m.add_regressor("holiday_flag")
    m.add_regressor("promo_flag")
    m.fit(_train_df)
    return m


df = load_data()


# ============================================================================
#  Tab 1 · 主看板
# ============================================================================
def render_dashboard_tab():
    st.sidebar.markdown("### 主看板参数")
    sku = st.sidebar.selectbox("选择 SKU", sorted(df["sku_id"].unique()), key="dash_sku")
    wh  = st.sidebar.selectbox("选择仓库", sorted(df["warehouse"].unique()), key="dash_wh")
    horizon = st.sidebar.slider("预测天数 (horizon)", 7, 56, 28, step=7,
                                help="预测未来多少天的需求，默认 28 天（4 周）")
    test_horizon = st.sidebar.slider("验证集天数 (test)", 7, 56, 28, step=7)
    review_period = st.sidebar.slider("评审周期 R (天)", 1, 14, 7)
    z_input = st.sidebar.select_slider(
        "服务水平 Z（用于自定义安全库存）",
        options=[1.28, 1.65, 1.96, 2.33],
        value=1.65,
        help="Z=1.28→90%, 1.65→95%, 1.96→97.5%, 2.33→99% 服务水平",
    )
    use_custom_ss = st.sidebar.checkbox(
        "用 Z×σ×√L 自定义安全库存", value=False
    )

    sub = (df[(df.sku_id == sku) & (df.warehouse == wh)]
           .sort_values("date").reset_index(drop=True))
    if len(sub) < test_horizon + 30:
        st.error("该 SKU/仓数据不足，请换一个组合")
        return

    ts = sub[["date", "demand_units", "holiday_flag", "promo_flag"]].rename(
        columns={"date": "ds", "demand_units": "y"})
    train = ts.iloc[:-test_horizon].copy()
    test  = ts.iloc[-test_horizon:].copy()

    with st.spinner("Prophet 训练中…"):
        model = fit_prophet(train, _cache_key=f"{sku}-{wh}-{test_horizon}")
        fut = model.make_future_dataframe(
            periods=test_horizon + max(0, horizon - test_horizon), freq="D")
        fut = fut.merge(ts[["ds", "holiday_flag", "promo_flag"]], on="ds", how="left")
        fut[["holiday_flag", "promo_flag"]] = fut[["holiday_flag", "promo_flag"]].fillna(0)
        fcst = model.predict(fut)

    fcst_test = fcst.iloc[len(train):len(train) + test_horizon].reset_index(drop=True)
    y_true = test["y"].values
    y_pred = np.maximum(fcst_test["yhat"].values, 0)
    mae  = mean_absolute_error(y_true, y_pred)
    mape = float(np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))))
    bias = float(np.mean(y_pred - y_true))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("验证集 MAE", f"{mae:.2f} units")
    c2.metric("验证集 MAPE", f"{mape:.1%}")
    c3.metric("预测偏差 Bias", f"{bias:+.2f}", delta="高估" if bias > 0 else "低估")
    c4.metric("提前期 L (天)", f"{int(sub['lead_time_days'].iloc[-1])}")

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(train["ds"], train["y"], color="#888", lw=1, label="历史实际")
    ax.plot(test["ds"], y_true, color="#1f77b4", lw=1.8, marker="o", ms=3, label="验证集实际")
    ax.plot(test["ds"], y_pred, color="#d62728", lw=1.8, label="验证集预测")
    ax.fill_between(test["ds"], fcst_test["yhat_lower"], fcst_test["yhat_upper"],
                    color="#d62728", alpha=0.18, label="95% 区间")
    if horizon > test_horizon:
        future_part = fcst.iloc[len(train) + test_horizon:].copy()
        ax.plot(future_part["ds"], np.maximum(future_part["yhat"], 0),
                color="#2ca02c", lw=1.8, ls="--", label="未来预测")
        ax.fill_between(future_part["ds"], future_part["yhat_lower"],
                        future_part["yhat_upper"], color="#2ca02c", alpha=0.12)
    ax.axvline(train["ds"].iloc[-1], color="k", ls="--", lw=0.8, alpha=0.6)
    ax.set_title(f"{sku} @ {wh} — Prophet 预测")
    ax.set_xlabel("date"); ax.set_ylabel("demand units")
    ax.legend(loc="upper left")
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("补货建议")
    snap_idx = len(sub) - test_horizon - 1
    snap = sub.iloc[snap_idx]
    L = int(snap["lead_time_days"])
    on_hand  = float(snap["on_hand_units"])
    on_order = float(snap["on_order_units"])
    SS_data  = float(snap["safety_stock_units"])

    sigma_d = float(train["y"].std())
    SS_custom = z_input * sigma_d * np.sqrt(L)
    SS_used = SS_custom if use_custom_ss else SS_data

    forecast_after_train = np.maximum(fcst["yhat"].values[len(train):], 0)
    fc_L = float(forecast_after_train[:L].sum())
    fc_R = float(forecast_after_train[:review_period].sum())
    ROP  = fc_L + SS_used
    S    = ROP + fc_R
    inv_pos = on_hand + on_order
    need_order = inv_pos < ROP
    suggest_qty = max(0.0, S - inv_pos) if need_order else 0.0

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("当前在手 on_hand", f"{on_hand:.0f}")
    m2.metric("在途 on_order",   f"{on_order:.0f}")
    m3.metric("安全库存 SS", f"{SS_used:.1f}",
              delta=f"{'自定义 Z='+str(z_input) if use_custom_ss else '来自数据'}")
    m4.metric("再订货点 ROP", f"{ROP:.1f}")
    m5.metric("订到点 S",     f"{S:.1f}")

    if need_order:
        st.success(f"⚠ 建议下单 **{suggest_qty:.0f} units**  "
                   f"（库存位置 {inv_pos:.0f} < ROP {ROP:.0f}）")
    else:
        st.info(f"✓ 暂不需要下单（库存位置 {inv_pos:.0f} ≥ ROP {ROP:.0f}）")

    detail = pd.DataFrame({
        "项目": ["决策日", "提前期 L (天)", "评审周期 R (天)",
                "提前期预测需求", "评审期预测需求",
                "安全库存", "ROP", "订到点 S",
                "在手 on_hand", "在途 on_order", "库存位置", "建议下单量"],
        "数值": [str(snap["date"].date()), str(L), str(review_period),
                f"{fc_L:.1f}", f"{fc_R:.1f}",
                f"{SS_used:.1f}", f"{ROP:.1f}", f"{S:.1f}",
                f"{on_hand:.0f}", f"{on_order:.0f}", f"{inv_pos:.0f}",
                f"{suggest_qty:.0f}"],
    })
    st.dataframe(detail, hide_index=True, width="stretch")

    with st.expander("查看 Prophet 趋势 / 周内季节性成分"):
        fig2 = model.plot_components(fcst)
        st.pyplot(fig2)

    with st.expander("查看历史原始数据（最近 30 天）"):
        st.dataframe(sub.tail(30), hide_index=True, width="stretch")


# ============================================================================
#  Tab 2 · AI 助手
# ============================================================================
def _render_chat_message(message: dict):
    """渲染 st.session_state.chat_log 里的一条历史消息。"""
    role = message["role"]
    with st.chat_message(role):
        for ev_type, payload in message["events"]:
            if ev_type == "text":
                st.markdown(payload)
            elif ev_type == "tool_use":
                st.info(
                    f"🔧 调用工具 `{payload['name']}` — "
                    f"参数 `{json.dumps(payload['input'], ensure_ascii=False)}`",
                    icon="🔧",
                )
            elif ev_type == "tool_result":
                tag = "❌" if payload["is_error"] else "✅"
                with st.expander(f"{tag} 工具返回：{payload['name']}", expanded=False):
                    try:
                        st.json(json.loads(payload["result_preview"].rstrip(".")))
                    except Exception:
                        st.code(payload["result_preview"])
            elif ev_type == "done":
                usage = payload.get("usage")
                if usage:
                    st.caption(
                        f"⏱ finish={payload['stop_reason']}  "
                        f"tokens: prompt={usage['prompt_tokens']} "
                        f"completion={usage['completion_tokens']} "
                        f"total={usage['total_tokens']}"
                    )


def render_agent_tab():
    st.sidebar.markdown("---")
    st.sidebar.markdown("### AI 助手设置")

    # API Key：优先环境变量，否则侧栏输入
    default_key = os.environ.get("ZHIPUAI_API_KEY", "")
    api_key = st.sidebar.text_input(
        "智谱 AI API Key（ZhipuAI）",
        value=default_key,
        type="password",
        help="从 https://open.bigmodel.cn/usercenter/apikeys 获取。"
             "也可设置环境变量 ZHIPUAI_API_KEY",
    )
    if st.sidebar.button("清空对话历史", key="clear_chat"):
        st.session_state.chat_log = []
        st.session_state.api_messages = make_initial_messages()  # 保留 system
        st.rerun()

    st.subheader("AI 助手 — 自然语言查预测、决策、下单")
    st.caption(
        "示例问题：\n"
        "- 数据里都有哪些 SKU？\n"
        "- SKU005 在 WH_A 仓库现在缺货风险高吗？\n"
        "- 帮我看下 SKU010 / WH_B 未来四周需要补多少货\n"
        "- 给 SKU003 在 WH_A 下 100 件的采购单"
    )

    # 初始化会话状态
    #   chat_log    : 给 UI 用的（每条记录是 {role, events: [...]}）
    #   api_messages: 给 GLM API 用的（标准 OpenAI messages 数组，
    #                 第一条是 system，含 assistant.tool_calls / tool.tool_call_id）
    if "chat_log" not in st.session_state:
        st.session_state.chat_log = []
    if "api_messages" not in st.session_state:
        st.session_state.api_messages = make_initial_messages()

    # 渲染历史
    for msg in st.session_state.chat_log:
        _render_chat_message(msg)

    # 输入框
    user_input = st.chat_input("输入你的问题，回车发送")
    if not user_input:
        return

    # 用户消息：UI + API 两份记录
    st.session_state.chat_log.append({
        "role": "user",
        "events": [("text", user_input)],
    })
    st.session_state.api_messages.append({"role": "user", "content": user_input})
    _render_chat_message(st.session_state.chat_log[-1])

    # 没有 API Key 直接报错
    if not api_key:
        with st.chat_message("assistant"):
            st.error("缺智谱 AI API Key。请在侧栏输入或设置 ZHIPUAI_API_KEY 环境变量。")
        st.session_state.chat_log.pop()
        st.session_state.api_messages.pop()
        return

    # 跑 Agent
    assistant_events: list[tuple[str, object]] = []
    with st.chat_message("assistant"):
        placeholder = st.empty()
        with st.spinner("Agent 思考中…"):
            try:
                client = get_client(api_key)
                # run_agent_turn 是生成器：边跑边 yield 事件，最终 mutate api_messages
                for ev_type, payload in run_agent_turn(client, st.session_state.api_messages):
                    assistant_events.append((ev_type, payload))
                    # 实时绘制（在同一个 placeholder 里反复重画整个 assistant 内容）
                    with placeholder.container():
                        _render_events_inline(assistant_events)
            except Exception as e:
                st.error(f"Agent 出错：{type(e).__name__}: {e}")
                with st.expander("点这里查看完整堆栈（复制贴回给我可定位问题）", expanded=False):
                    st.code(traceback.format_exc(), language="text")
                # 把这一轮失败的 user 消息也回滚，避免下次重发时上下文不完整
                st.session_state.chat_log.pop()
                st.session_state.api_messages.pop()
                return

    st.session_state.chat_log.append({
        "role": "assistant",
        "events": assistant_events,
    })


def _render_events_inline(events: list[tuple[str, object]]):
    """实时渲染 agent 事件流（与历史渲染逻辑一致）。"""
    for ev_type, payload in events:
        if ev_type == "text":
            st.markdown(payload)
        elif ev_type == "tool_use":
            st.info(
                f"🔧 调用工具 `{payload['name']}` — "
                f"参数 `{json.dumps(payload['input'], ensure_ascii=False)}`",
                icon="🔧",
            )
        elif ev_type == "tool_result":
            tag = "❌" if payload["is_error"] else "✅"
            with st.expander(f"{tag} 工具返回：{payload['name']}", expanded=False):
                try:
                    st.json(json.loads(payload["result_preview"].rstrip(".")))
                except Exception:
                    st.code(payload["result_preview"])
        elif ev_type == "done":
            usage = payload.get("usage")
            if usage:
                st.caption(
                    f"⏱ finish={payload['stop_reason']}  "
                    f"tokens: prompt={usage['prompt_tokens']} "
                    f"completion={usage['completion_tokens']} "
                    f"total={usage['total_tokens']}"
                )


# ============================================================================
#  主入口：两个标签页
# ============================================================================
tab1, tab2 = st.tabs(["📊 主看板", "🤖 AI 助手"])
with tab1:
    render_dashboard_tab()
with tab2:
    render_agent_tab()
