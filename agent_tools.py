"""
=================================================================
 Agent 工具层：把已有的 Prophet + ROP 计算逻辑封装成可调用工具
-----------------------------------------------------------------
 5 个工具：
   1. list_skus               — 列出所有 SKU/仓库
   2. get_inventory_status    — 查询当前库存状态
   3. forecast_demand         — Prophet 预测未来 N 天需求
   4. compute_replenishment   — 算 ROP + 建议补货量
   5. place_order             — 沙盒模拟下单（仅打印 + 写日志）

 设计原则：
   · 每个工具都是普通 Python 函数，返回 dict（方便 JSON 序列化给 LLM）
   · 数据集和 Prophet 预测做了内存缓存，避免 LLM 多次提问时反复重训
   · TOOL_SCHEMAS_OPENAI 是 GLM/OpenAI 格式工具定义，TOOL_DISPATCH 是名字→函数映射
   · 保留 TOOL_SCHEMAS（Anthropic 原始格式）作为单一事实来源，OpenAI 格式从它派生
=================================================================
"""
from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

# Windows 子进程 stdout/stderr 默认编码可能不是 UTF-8，print 中文会爆 'ascii' codec
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")

from prophet import Prophet  # noqa: E402

# ---------- 配置 ----------
DATA_PATH = "inventory_replenishment_timeseries_10000.csv"
TEST_HORIZON = 28
REVIEW_PERIOD = 7
SANDBOX_LOG_PATH = Path("outputs") / "sandbox_orders.log"
SANDBOX_LOG_PATH.parent.mkdir(exist_ok=True)


# ---------- 数据 + 模型缓存（避免每次工具调用都重训）----------
_DATA: pd.DataFrame | None = None
_FORECAST_CACHE: dict[tuple[str, str, int], dict] = {}


def _load_data() -> pd.DataFrame:
    global _DATA
    if _DATA is None:
        _DATA = pd.read_csv(DATA_PATH, parse_dates=["date"])
    return _DATA


def _get_sub(sku_id: str, warehouse: str) -> pd.DataFrame:
    df = _load_data()
    sub = (df[(df.sku_id == sku_id) & (df.warehouse == warehouse)]
           .sort_values("date").reset_index(drop=True))
    if len(sub) == 0:
        raise ValueError(f"未找到 SKU={sku_id} 仓库={warehouse} 的数据")
    return sub


def _fit_and_forecast(sku_id: str, warehouse: str, horizon: int = 28) -> dict:
    """训练 Prophet 并预测未来 horizon 天。带内存缓存。"""
    key = (sku_id, warehouse, horizon)
    if key in _FORECAST_CACHE:
        return _FORECAST_CACHE[key]

    sub = _get_sub(sku_id, warehouse)
    ts = sub[["date", "demand_units", "holiday_flag", "promo_flag"]].rename(
        columns={"date": "ds", "demand_units": "y"})
    train = ts.iloc[:-TEST_HORIZON]
    test  = ts.iloc[-TEST_HORIZON:]

    m = Prophet(yearly_seasonality=False, weekly_seasonality=True,
                daily_seasonality=False, interval_width=0.95)
    m.add_regressor("holiday_flag")
    m.add_regressor("promo_flag")
    m.fit(train)

    # Prophet 的 future 需要覆盖：验证段 (TEST_HORIZON 天) + 真正未来段 (horizon 天)
    future = m.make_future_dataframe(periods=TEST_HORIZON + horizon, freq="D")
    future = future.merge(ts[["ds", "holiday_flag", "promo_flag"]], on="ds", how="left")
    future[["holiday_flag", "promo_flag"]] = future[["holiday_flag", "promo_flag"]].fillna(0)
    fcst = m.predict(future)

    # 验证段（训练截止后的 TEST_HORIZON 天，对应 ts.iloc[-TEST_HORIZON:]）
    fcst_test = fcst.iloc[len(train):len(train) + TEST_HORIZON]
    y_true = test["y"].values
    y_pred = np.maximum(fcst_test["yhat"].values, 0)
    mae = float(mean_absolute_error(y_true, y_pred))
    rel_mae = mae / max(float(y_true.mean()), 1e-6)

    # 真正未来段（验证段之后的 horizon 天）
    future_part = fcst.iloc[len(train) + TEST_HORIZON:]
    future_dates = future_part["ds"].dt.strftime("%Y-%m-%d").tolist()
    future_yhat  = np.maximum(future_part["yhat"].values, 0).round(1).tolist()
    future_lo    = future_part["yhat_lower"].values.round(1).tolist()
    future_hi    = future_part["yhat_upper"].values.round(1).tolist()

    result = dict(
        validation_mae=round(mae, 2),
        validation_relative_mae=round(rel_mae, 4),
        validation_y_mean=round(float(y_true.mean()), 2),
        future_horizon=horizon,
        future_dates=future_dates,
        future_forecast=future_yhat,
        future_lower=future_lo,
        future_upper=future_hi,
        forecast_sum=round(float(sum(future_yhat)), 1),
    )
    _FORECAST_CACHE[key] = result
    return result


# ======================== 工具 1 ========================
def list_skus(warehouse: str | None = None) -> dict:
    """列出所有 SKU 与仓库组合。可选按仓库过滤。"""
    df = _load_data()
    if warehouse:
        df = df[df.warehouse == warehouse]
    combos = (df.groupby(["sku_id", "warehouse"])
                .agg(records=("date", "count"),
                     avg_daily_demand=("demand_units", "mean"),
                     lead_time=("lead_time_days", "first"))
                .round(2).reset_index())
    return dict(
        total=len(combos),
        warehouses=sorted(df["warehouse"].unique().tolist()),
        skus=combos.to_dict(orient="records"),
    )


# ======================== 工具 2 ========================
def get_inventory_status(sku_id: str, warehouse: str) -> dict:
    """查询某 SKU 在某仓库的最新库存状态。"""
    sub = _get_sub(sku_id, warehouse)
    snap_idx = len(sub) - TEST_HORIZON - 1
    snap = sub.iloc[snap_idx]
    hist = sub.iloc[:snap_idx + 1]
    return dict(
        sku_id=sku_id,
        warehouse=warehouse,
        snapshot_date=str(snap["date"].date()),
        on_hand=int(snap["on_hand_units"]),
        on_order=int(snap["on_order_units"]),
        inventory_position=int(snap["on_hand_units"] + snap["on_order_units"]),
        safety_stock=round(float(snap["safety_stock_units"]), 1),
        lead_time_days=int(snap["lead_time_days"]),
        service_level_target=float(snap["service_level_target"]),
        avg_daily_demand=round(float(hist["demand_units"].mean()), 2),
        std_daily_demand=round(float(hist["demand_units"].std()), 2),
        historical_stockout_days=int(hist["stockout"].sum()),
        historical_fill_rate=round(float(hist["fill_rate"].mean()), 4),
    )


# ======================== 工具 3 ========================
def forecast_demand(sku_id: str, warehouse: str, horizon: int = 28) -> dict:
    """用 Prophet 预测未来 horizon 天的需求。返回验证集指标 + 预测曲线。"""
    horizon = max(1, min(int(horizon), 90))  # 安全范围
    fc = _fit_and_forecast(sku_id, warehouse, horizon)
    return dict(
        sku_id=sku_id,
        warehouse=warehouse,
        model="Prophet (weekly seasonality + holiday/promo regressors)",
        validation_mae=fc["validation_mae"],
        validation_relative_mae_pct=round(fc["validation_relative_mae"] * 100, 2),
        validation_y_mean=fc["validation_y_mean"],
        horizon_days=horizon,
        forecast_total=fc["forecast_sum"],
        forecast_avg_per_day=round(fc["forecast_sum"] / horizon, 2),
        first_7_days=fc["future_forecast"][:7],
        last_7_days=fc["future_forecast"][-7:] if horizon >= 7 else fc["future_forecast"],
        future_dates_sample=[fc["future_dates"][0], fc["future_dates"][-1]],
    )


# ======================== 工具 4 ========================
def compute_replenishment(sku_id: str, warehouse: str,
                          review_period: int = 7) -> dict:
    """计算 ROP（再订货点）、订到点 S、建议下单量。"""
    inv = get_inventory_status(sku_id, warehouse)
    L = inv["lead_time_days"]
    SS = inv["safety_stock"]
    fc = _fit_and_forecast(sku_id, warehouse, horizon=max(L + review_period, 28))
    forecast_series = fc["future_forecast"]

    lead_demand = round(float(sum(forecast_series[:L])), 1)
    review_demand = round(float(sum(forecast_series[:review_period])), 1)
    rop = round(lead_demand + SS, 1)
    order_up_to_S = round(rop + review_demand, 1)
    inv_pos = inv["inventory_position"]
    need_order = inv_pos < rop
    suggest_qty = round(max(0.0, order_up_to_S - inv_pos), 0) if need_order else 0

    return dict(
        sku_id=sku_id,
        warehouse=warehouse,
        lead_time_days=L,
        review_period_days=review_period,
        safety_stock=SS,
        on_hand=inv["on_hand"],
        on_order=inv["on_order"],
        inventory_position=inv_pos,
        forecast_lead_time_demand=lead_demand,
        forecast_review_period_demand=review_demand,
        reorder_point=rop,
        order_up_to_S=order_up_to_S,
        need_order=need_order,
        suggested_order_qty=int(suggest_qty),
        rationale=(
            f"库存位置 {inv_pos} {'<' if need_order else '≥'} ROP {rop}（提前期需求 "
            f"{lead_demand} + 安全库存 {SS}）"
        ),
    )


# ======================== 工具 5（沙盒）========================
def place_order(sku_id: str, quantity: int, warehouse: str = "WH_A",
                note: str = "") -> dict:
    """[沙盒] 模拟下单。仅打印日志 + 追加到 outputs/sandbox_orders.log，不会真正发出采购单。"""
    quantity = int(quantity)
    if quantity <= 0:
        return dict(success=False, message=f"数量必须为正整数，收到 {quantity}")

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[SANDBOX 沙盒][{ts}] 已下单 {sku_id} @ {warehouse}  {quantity} 件"
    if note:
        line += f"  （备注：{note}）"
    # print 防御：哪怕 stdout 编码异常，也不能让一次"打日志"把整个 Agent 流程拖垮
    try:
        print(line)
    except UnicodeEncodeError:
        sys.stderr.write(line.encode("utf-8", errors="replace").decode("utf-8") + "\n")
    # 文件落盘始终用 UTF-8，可靠
    with open(SANDBOX_LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")

    return dict(
        success=True,
        sandbox=True,
        order_id=f"SBX-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        sku_id=sku_id, warehouse=warehouse, quantity=quantity,
        timestamp=ts,
        message=f"沙盒环境已记录订单：SKU {sku_id} × {quantity} 件 @ {warehouse}。"
                f"此为演示用模拟下单，未触发真实采购流程。",
    )


# ===================================================================
#  工具描述（JSON Schema）- 单一事实来源
#  这些定义告诉模型每个工具的作用 + 参数。模型只会根据 description 来决策。
#  下方 TOOL_SCHEMAS_OPENAI 由此自动派生为 GLM / OpenAI 兼容格式。
# ===================================================================
TOOL_SCHEMAS = [
    {
        "name": "list_skus",
        "description": "列出所有可用的 SKU 和仓库组合，包含每条序列的记录数、平均日需求、提前期。"
                       "当用户问'有哪些 SKU'、'哪些仓库'、'数据规模'时使用。",
        "input_schema": {
            "type": "object",
            "properties": {
                "warehouse": {
                    "type": "string",
                    "description": "可选：按仓库筛选（如 WH_A 或 WH_B）。不填则返回全部。",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_inventory_status",
        "description": "查询指定 SKU 在指定仓库的最新库存状态：在手、在途、安全库存、提前期、"
                       "历史平均日需求、历史缺货天数、Fill Rate 等。"
                       "决策'要不要补货'前几乎总要先调用这个。",
        "input_schema": {
            "type": "object",
            "properties": {
                "sku_id": {"type": "string", "description": "SKU 编号，如 SKU001"},
                "warehouse": {"type": "string", "description": "仓库编号，WH_A 或 WH_B"},
            },
            "required": ["sku_id", "warehouse"],
        },
    },
    {
        "name": "forecast_demand",
        "description": "用 Prophet 模型预测某 SKU 在某仓库未来 horizon 天的需求量，并返回验证集 MAE。"
                       "用于回答'未来 4 周需求多少'、'预测准不准'之类的问题。",
        "input_schema": {
            "type": "object",
            "properties": {
                "sku_id": {"type": "string", "description": "SKU 编号"},
                "warehouse": {"type": "string", "description": "仓库编号"},
                "horizon": {
                    "type": "integer",
                    "description": "预测天数，1~90 之间，默认 28（4 周）",
                    "default": 28,
                },
            },
            "required": ["sku_id", "warehouse"],
        },
    },
    {
        "name": "compute_replenishment",
        "description": "结合 Prophet 预测和当前库存，计算动态再订货点 ROP、订到点 S，并给出建议补货量。"
                       "用于回答'要补多少'、'什么时候补'。**会自动调用 forecast_demand 和 "
                       "get_inventory_status**，所以可以直接调用这个工具，不必先单独问预测和库存。",
        "input_schema": {
            "type": "object",
            "properties": {
                "sku_id": {"type": "string"},
                "warehouse": {"type": "string"},
                "review_period": {
                    "type": "integer",
                    "description": "评审周期（天），默认 7（每周一次）",
                    "default": 7,
                },
            },
            "required": ["sku_id", "warehouse"],
        },
    },
    {
        "name": "place_order",
        "description": "【沙盒模拟】下采购单。仅会写日志到 outputs/sandbox_orders.log 并打印，不会触发真实采购。"
                       "调用前必须先用 compute_replenishment 计算建议数量，并在回复中向用户确认数量"
                       "（'我建议下 100 件，是否执行？'），用户明确同意后再调用。",
        "input_schema": {
            "type": "object",
            "properties": {
                "sku_id":    {"type": "string", "description": "SKU 编号"},
                "warehouse": {"type": "string", "description": "仓库编号"},
                "quantity":  {"type": "integer", "description": "下单数量（正整数）"},
                "note":      {"type": "string",  "description": "可选备注"},
            },
            "required": ["sku_id", "warehouse", "quantity"],
        },
    },
]


# 名字 → 函数 的分发表
TOOL_DISPATCH = {
    "list_skus":              list_skus,
    "get_inventory_status":   get_inventory_status,
    "forecast_demand":        forecast_demand,
    "compute_replenishment":  compute_replenishment,
    "place_order":            place_order,
}


# ===================================================================
#  OpenAI / GLM 兼容格式的工具定义
#  ZhipuAI（智谱 GLM）走 OpenAI 协议：tool 定义要包一层 type/function。
#  parameters 字段 = input_schema 内容（JSON Schema 标准）。
# ===================================================================
TOOL_SCHEMAS_OPENAI = [
    {
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["input_schema"],
        },
    }
    for t in TOOL_SCHEMAS
]


def dispatch_tool(name: str, arguments: dict) -> dict:
    """根据名字调用对应工具，返回 dict 结果。"""
    if name not in TOOL_DISPATCH:
        raise ValueError(f"未知工具：{name}")
    fn = TOOL_DISPATCH[name]
    return fn(**arguments)


# ---------- 自检（直接 python agent_tools.py 可跑）----------
if __name__ == "__main__":
    print("=== 工具自检 ===")
    print(json.dumps(list_skus("WH_A"), ensure_ascii=False, indent=2)[:400], "...")
    print("\n--- get_inventory_status(SKU001, WH_A) ---")
    print(json.dumps(get_inventory_status("SKU001", "WH_A"), ensure_ascii=False, indent=2))
    print("\n--- compute_replenishment(SKU001, WH_A) ---")
    print(json.dumps(compute_replenishment("SKU001", "WH_A"), ensure_ascii=False, indent=2))
    print("\n--- place_order(SKU001, 100, WH_A) ---")
    print(json.dumps(place_order("SKU001", 100, "WH_A", note="自检"),
                     ensure_ascii=False, indent=2))
