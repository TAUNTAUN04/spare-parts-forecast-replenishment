"""
============================================================================
 智能备件需求预测与补货决策 — 主实验脚本（MVP）
----------------------------------------------------------------------------
 数据：inventory_replenishment_timeseries_10000.csv  (20 SKU × 2 仓库 × 250 天)
 流程：
   SECTION 1  数据加载 + EDA
   SECTION 2  单 SKU·单仓 Prophet 教学演示（讲透流程 + 出图）
   SECTION 3  批量 Prophet 验证（40 条序列） → 汇总 MAE / MAPE / sMAPE
   SECTION 4  动态 ROP（Reorder Point）补货建议
   SECTION 5  策略对比仿真：固定阈值 vs 动态 ROP（28 天测试期）
   SECTION 6  导出 CSV + PNG，并打印关键 KPI 汇总
----------------------------------------------------------------------------
 核心产出指标：
   - SECTION 3 末 "全体平均 相对MAE"   → 预测精度 KPI
   - SECTION 5 末 "缺货风险相对降低"   → 补货策略对比的核心收益
   - SECTION 6 末 outputs/ 下完整文件 → CSV + PNG 一站式产出
============================================================================
"""

# ============ SECTION 0 · 环境与全局参数 ============
import os
import sys
import logging
import warnings
from datetime import datetime
from pathlib import Path

# ---- 自动 tee：所有 print/stderr 同步写到 run_logs/<时间戳>.txt ----
# 命名格式：YYYYMMDDHHMM（如 202605131518）。若同一分钟内重跑会被覆盖。
LOG_DIR = Path("run_logs")
LOG_DIR.mkdir(exist_ok=True)
RUN_ID = datetime.now().strftime("%Y%m%d%H%M")
LOG_FILE = LOG_DIR / f"{RUN_ID}.txt"


class _Tee:
    """同时把写入分发到多个流（终端 + 日志文件）。"""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
            except Exception:
                pass

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass


_log_fh = open(LOG_FILE, "w", encoding="utf-8", buffering=1)  # 行缓冲，实时落盘
sys.stdout = _Tee(sys.__stdout__, _log_fh)
sys.stderr = _Tee(sys.__stderr__, _log_fh)
print(f"[run log] 本次输出同步写入 → {LOG_FILE.resolve()}")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Prophet 输出比较吵，先静音
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")
from prophet import Prophet  # noqa: E402

# ---- 全局参数（可在此调参，所有 SECTION 都引用这些）----
DATA_PATH        = "inventory_replenishment_timeseries_10000.csv"
OUTPUT_DIR       = "outputs"
FORECAST_HORIZON = 28        # 预测未来 4 周（28 天）
TEST_HORIZON     = 28        # 用最后 28 天做验证
Z_SCORE          = 1.65      # 95% 服务水平对应的正态分位数（备用：自行计算安全库存时用）
REVIEW_PERIOD    = 7         # 周补单：每 7 天评审一次库存是否下单

# ---- 基线策略（"传统经验式" — 管理员拍脑袋的全局固定阈值）----
# 现实里许多中小企业并不为每个 SKU 单独算 ROP，而是用一个全局统一的数字。
# 这是 AI/数据驱动方法真正要替代的对象。
GLOBAL_THRESHOLD = 100   # 全局再订货阈值 T（不分 SKU、不考虑 lead time）
GLOBAL_ORDER_QTY = 200   # 全局固定下单量 Q

os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def banner(title: str) -> None:
    """打印分节横幅。"""
    line = "=" * 76
    print("\n" + line)
    print(f"  {title}")
    print(line)


# ============ SECTION 1 · 数据加载与 EDA ============
banner("SECTION 1 · 数据加载与初步探索")

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
print(f"[基本信息] 行数 = {len(df):,}    列数 = {df.shape[1]}")
print(f"[时间范围] {df['date'].min().date()}  →  {df['date'].max().date()}")
print(f"[规模]    SKU 数 = {df['sku_id'].nunique()}    仓库数 = {df['warehouse'].nunique()}")
print(f"[策略列]  policy = {df['policy'].unique()}")

# --- 缺失值 ---
miss = df.isna().sum()
print(f"\n[缺失值] 共 {int(miss.sum())} 个；按列：")
print(miss[miss > 0] if miss.sum() else "  ✓ 无缺失")

# --- 需求统计（核心列：demand_units）---
print("\n[需求统计] demand_units")
print(df["demand_units"].describe().round(2).to_string())

# --- 缺货情况 ---
n_stockout = int(df["stockout"].sum())
print(f"\n[原始数据缺货事件] 历史缺货天数 = {n_stockout}  "
      f"占比 = {df['stockout'].mean():.2%}  "
      f"原始 Fill Rate 平均 = {df['fill_rate'].mean():.2%}")

# --- SKU/仓 维度概览 ---
sku_summary = (df.groupby(["sku_id", "warehouse"])
                 .agg(days=("date", "count"),
                      mean_demand=("demand_units", "mean"),
                      std_demand=("demand_units", "std"),
                      lead_time=("lead_time_days", "first"),
                      safety_stock=("safety_stock_units", "mean"),
                      stockout_days=("stockout", "sum"))
                 .round(2).reset_index())
print(f"\n[各 SKU·仓概览]  形状 = {sku_summary.shape}")
print(sku_summary.head(8).to_string(index=False))
print("  …（仅展示前 8 行；完整结果将存盘）")
sku_summary.to_csv(f"{OUTPUT_DIR}/sku_warehouse_summary.csv", index=False)

# --- 简单可视化：整体每日需求曲线（聚合）+ 各 SKU 平均需求 ---
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
daily_total = df.groupby("date")["demand_units"].sum()
axes[0].plot(daily_total.index, daily_total.values, color="#2c7fb8", lw=1)
axes[0].set_title("全公司每日总需求")
axes[0].set_xlabel("date"); axes[0].set_ylabel("units")

sku_mean = df.groupby("sku_id")["demand_units"].mean().sort_values(ascending=False)
axes[1].bar(sku_mean.index, sku_mean.values, color="#fdae6b")
axes[1].set_title("各 SKU 平均日需求")
axes[1].set_xlabel("sku"); axes[1].set_ylabel("mean demand")
axes[1].tick_params(axis="x", rotation=60)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/eda_overview.png", dpi=120)
plt.close()
print(f"\n[已保存] {OUTPUT_DIR}/eda_overview.png  (EDA 概览图)")


# ============ SECTION 2 · 单 SKU 教学演示 ============
banner("SECTION 2 · Prophet 教学演示（SKU001 @ WH_A）")

demo_sku, demo_wh = "SKU001", "WH_A"
sub = (df[(df.sku_id == demo_sku) & (df.warehouse == demo_wh)]
       .sort_values("date").reset_index(drop=True))
print(f"取 {demo_sku} @ {demo_wh}：共 {len(sub)} 天")

# --- Step 1: 把列名改成 Prophet 要求的 ds/y ---
ts = sub[["date", "demand_units", "holiday_flag", "promo_flag"]].rename(
    columns={"date": "ds", "demand_units": "y"})
print("\n[Step 1] Prophet 输入要求列名 'ds' (datestamp) 与 'y' (target)，已重命名")
print(ts.head(3).to_string(index=False))

# --- Step 2: 划分训练/验证 ---
train = ts.iloc[:-TEST_HORIZON].copy()
test  = ts.iloc[-TEST_HORIZON:].copy()
print(f"\n[Step 2] 训练集 = 前 {len(train)} 天  |  验证集 = 后 {len(test)} 天 (= 4 周)")
print(f"   训练区间：{train.ds.min().date()}  →  {train.ds.max().date()}")
print(f"   验证区间：{test.ds.min().date()}   →  {test.ds.max().date()}")

# --- Step 3: 拟合 Prophet（带额外回归变量 holiday/promo） ---
print("\n[Step 3] 训练 Prophet …")
print("         · weekly_seasonality=True  → 捕捉周内周期（周一 vs 周日不同）")
print("         · 额外回归变量 holiday_flag / promo_flag → 节假日 + 促销对需求的脉冲")
print("         · interval_width=0.95     → 输出 95% 预测区间，方便用于安全库存推算")

m = Prophet(
    yearly_seasonality=False,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode="additive",
    interval_width=0.95,
)
m.add_regressor("holiday_flag")
m.add_regressor("promo_flag")
m.fit(train)
print("   ✓ 训练完成")

# --- Step 4: 预测未来 28 天 ---
future = m.make_future_dataframe(periods=TEST_HORIZON, freq="D")
future = future.merge(ts[["ds", "holiday_flag", "promo_flag"]], on="ds", how="left")
future[["holiday_flag", "promo_flag"]] = future[["holiday_flag", "promo_flag"]].fillna(0)
fcst = m.predict(future)
fcst_test = fcst.tail(TEST_HORIZON).reset_index(drop=True)

print("\n[Step 4] Prophet 输出关键列：")
print(fcst_test[["ds", "yhat", "yhat_lower", "yhat_upper",
                 "trend", "weekly"]].head(7).round(2).to_string(index=False))
print("   · yhat        = 点预测")
print("   · yhat_lower/upper = 95% 预测区间")
print("   · trend       = 长期趋势成分")
print("   · weekly      = 周内季节性成分（周一 vs 周日的相对偏移）")

# --- Step 5: 评估 ---
y_true = test["y"].values
y_pred = np.maximum(fcst_test["yhat"].values, 0)   # 需求不会为负，裁剪
mae    = mean_absolute_error(y_true, y_pred)
rmse   = float(np.sqrt(mean_squared_error(y_true, y_pred)))
mape   = float(np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))))
smape  = float(np.mean(2 * np.abs(y_true - y_pred) /
                       np.where((np.abs(y_true) + np.abs(y_pred)) == 0, 1,
                                np.abs(y_true) + np.abs(y_pred))))
bias   = float(np.mean(y_pred - y_true))

print(f"\n[Step 5] 验证集指标（{TEST_HORIZON} 天）")
print(f"   MAE   = {mae:.2f}  units      ← 平均每天预测偏差（越小越好）")
print(f"   RMSE  = {rmse:.2f}  units     ← 大偏差被放大（对异常更敏感）")
print(f"   MAPE  = {mape:.2%}            ← 相对百分比误差")
print(f"   sMAPE = {smape:.2%}           ← 对称版 MAPE，y=0 时更稳健")
print(f"   Bias  = {bias:+.2f}            ← >0 表示系统性高估，<0 表示低估")
print(f"   验证集 y 均值 = {y_true.mean():.2f}  →  相对 MAE = {mae/y_true.mean():.2%}")

# --- Step 6: 出图 ---
fig, ax = plt.subplots(figsize=(13, 4.2))
ax.plot(train["ds"], train["y"], label="历史实际", color="#888", lw=1)
ax.plot(test["ds"], y_true, label="验证集实际", color="#1f77b4", lw=1.8, marker="o", ms=3)
ax.plot(test["ds"], y_pred, label="Prophet 预测", color="#d62728", lw=1.8)
ax.fill_between(test["ds"], fcst_test["yhat_lower"], fcst_test["yhat_upper"],
                color="#d62728", alpha=0.18, label="95% 预测区间")
ax.axvline(train["ds"].iloc[-1], color="k", ls="--", lw=0.8, alpha=0.6)
ax.set_title(f"{demo_sku} @ {demo_wh} — Prophet 4 周预测 vs 实际")
ax.set_xlabel("date"); ax.set_ylabel("demand units"); ax.legend(loc="upper left")
plt.tight_layout()
demo_png = f"{OUTPUT_DIR}/demo_forecast_{demo_sku}_{demo_wh}.png"
plt.savefig(demo_png, dpi=120); plt.close()
print(f"\n[Step 6] 已保存示意图 → {demo_png}")


# ============ SECTION 3 · 批量 Prophet 验证（40 条序列）============
banner("SECTION 3 · 批量预测验证（20 SKU × 2 仓库）")

metrics_rows = []
# 保存每条序列的预测，给 SECTION 4/5 复用，避免重训
forecast_storage: dict[tuple[str, str], dict] = {}

groups = list(df.groupby(["sku_id", "warehouse"]))
print(f"待处理序列数：{len(groups)}（约耗时 30~90 秒）")

for i, ((sku, wh), g) in enumerate(groups, 1):
    g = g.sort_values("date").reset_index(drop=True)
    ts_i = g[["date", "demand_units", "holiday_flag", "promo_flag"]].rename(
        columns={"date": "ds", "demand_units": "y"})
    tr = ts_i.iloc[:-TEST_HORIZON]
    te = ts_i.iloc[-TEST_HORIZON:]
    try:
        mi = Prophet(yearly_seasonality=False, weekly_seasonality=True,
                     daily_seasonality=False, interval_width=0.95)
        mi.add_regressor("holiday_flag")
        mi.add_regressor("promo_flag")
        mi.fit(tr)

        fut = mi.make_future_dataframe(periods=TEST_HORIZON, freq="D")
        fut = fut.merge(ts_i[["ds", "holiday_flag", "promo_flag"]], on="ds", how="left")
        fut[["holiday_flag", "promo_flag"]] = fut[["holiday_flag", "promo_flag"]].fillna(0)
        fc = mi.predict(fut)
        fc_te = fc.tail(TEST_HORIZON).reset_index(drop=True)

        yt = te["y"].values
        yp = np.maximum(fc_te["yhat"].values, 0)
        mae_i = mean_absolute_error(yt, yp)
        mape_i = float(np.mean(np.abs((yt - yp) / np.where(yt == 0, 1, yt))))
        smape_i = float(np.mean(2 * np.abs(yt - yp) /
                                np.where((np.abs(yt) + np.abs(yp)) == 0, 1,
                                         np.abs(yt) + np.abs(yp))))
        bias_i = float(np.mean(yp - yt))
        # 朴素 baseline：用训练集最后 28 天均值作为常数预测
        naive_p = np.repeat(tr["y"].tail(28).mean(), TEST_HORIZON)
        naive_mae = mean_absolute_error(yt, naive_p)

        metrics_rows.append(dict(
            sku=sku, warehouse=wh,
            y_mean=float(yt.mean()), y_std=float(yt.std()),
            mae=mae_i, mape=mape_i, smape=smape_i, bias=bias_i,
            naive_mae=naive_mae,
            improvement_vs_naive=(naive_mae - mae_i) / max(naive_mae, 1e-6),
        ))
        forecast_storage[(sku, wh)] = dict(
            train=tr, test=te,
            test_dates=te["ds"].values, y_true=yt, y_pred=yp,
            yhat_lower=fc_te["yhat_lower"].values,
            yhat_upper=fc_te["yhat_upper"].values,
        )
        if i % 5 == 0 or i == len(groups):
            print(f"  [{i:2d}/{len(groups)}] {sku}/{wh}  "
                  f"MAE={mae_i:5.2f}  MAPE={mape_i:6.2%}  bias={bias_i:+.2f}")
    except Exception as e:
        print(f"  [warn] {sku}/{wh} 失败：{e}")

metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv(f"{OUTPUT_DIR}/validation_metrics_per_sku.csv", index=False)

# --- 汇总 ---
overall_mae   = metrics_df["mae"].mean()
overall_mape  = metrics_df["mape"].mean()
overall_smape = metrics_df["smape"].mean()
rel_mae       = (metrics_df["mae"] / metrics_df["y_mean"]).mean()
naive_mae_avg = metrics_df["naive_mae"].mean()
gain          = (naive_mae_avg - overall_mae) / max(naive_mae_avg, 1e-6)

print("\n[SECTION 3 汇总]")
print(metrics_df[["mae", "mape", "smape", "bias", "naive_mae"]].describe().round(3).to_string())
print(f"\n>>> 全体平均 MAE            = {overall_mae:.2f} units")
print(f">>> 全体平均 MAPE           = {overall_mape:.2%}")
print(f">>> 全体平均 sMAPE          = {overall_smape:.2%}")
print(f">>> 全体平均 相对MAE        = {rel_mae:.2%}     ← ★ 关键 KPI：预测精度")
print(f">>> Prophet 相对 naive 改善 = {gain:+.2%}     ← 衡量 ML 模型相对最朴素基线的增益")


# ============ SECTION 4 · 动态 ROP 与补货建议 ============
banner("SECTION 4 · 动态 ROP + 推荐补货量")
print("""\
理论速览：
  · 再订货点 ROP = 提前期需求 + 安全库存
        Reorder Point = Lead_Time_Demand + Safety_Stock
  · 订到点 S  (order-up-to level) = ROP + 评审期需求
  · 库存位置 Inventory Position = on_hand + on_order
  · 触发条件：Inventory Position < ROP → 下单 Q = S − Inventory Position
""")

recommend_rows = []
for (sku, wh), pack in forecast_storage.items():
    s_df = (df[(df.sku_id == sku) & (df.warehouse == wh)]
            .sort_values("date").reset_index(drop=True))
    # 决策时点 = 训练集最后一天的快照
    snap_idx = len(s_df) - TEST_HORIZON - 1
    snap = s_df.iloc[snap_idx]
    L  = int(snap["lead_time_days"])
    SS = float(snap["safety_stock_units"])
    on_hand  = float(snap["on_hand_units"])
    on_order = float(snap["on_order_units"])

    yp = pack["y_pred"]
    fc_L = float(yp[:L].sum())                  # 提前期需求预测
    R    = REVIEW_PERIOD
    fc_R = float(yp[:R].sum())                  # 评审期需求预测
    ROP  = fc_L + SS
    S    = ROP + fc_R
    inv_pos = on_hand + on_order
    suggest_qty = max(0.0, S - inv_pos) if inv_pos < ROP else 0.0

    recommend_rows.append(dict(
        sku=sku, warehouse=wh,
        lead_time=L, safety_stock=round(SS, 1),
        on_hand=round(on_hand, 1), on_order=round(on_order, 1),
        inventory_position=round(inv_pos, 1),
        forecast_lead_demand=round(fc_L, 1),
        ROP=round(ROP, 1), order_up_to_S=round(S, 1),
        need_order=bool(inv_pos < ROP),
        suggest_qty=round(suggest_qty, 0),
    ))

rec_df = pd.DataFrame(recommend_rows)
rec_df.to_csv(f"{OUTPUT_DIR}/replenishment_recommendation.csv", index=False)
print(rec_df.head(15).to_string(index=False))
print("  …（完整结果见 outputs/replenishment_recommendation.csv）")
print(f"\n需补货 SKU/仓 数：{int(rec_df['need_order'].sum())} / {len(rec_df)}")
print(f"总建议下单量    ：{rec_df['suggest_qty'].sum():.0f} units")


# ============ SECTION 5 · 策略对比仿真（28 天测试期）============
banner("SECTION 5 · 三种补货策略对比（28 天仿真）")
print(f"""\
对比设计：

策略 A  ·  全局固定阈值（"管理员拍脑袋"基线）
  · 全 SKU 同一个数字：T = {GLOBAL_THRESHOLD}, Q = {GLOBAL_ORDER_QTY}
  · 不分 SKU、不考虑 lead_time、不考虑需求大小
  · 这是大量中小企业的真实做法 → AI 替代的就是它

策略 B  ·  每 SKU 历史均值（"数据看了一眼"基线）
  · ROP = avg_daily × L（每 SKU 自己的均值，但仍然是"静态"）
  · 下单量 Q = avg_daily × (L + R)
  · 体现"分 SKU 管理"的价值，但没有时序感知

策略 C  ·  Prophet 动态 ROP（本项目主力方案）
  · ROP = Prophet 预测未来 L 天需求 + 安全库存
  · 订到点 S = ROP + Prophet 预测未来 R 天需求
  · 同时具备"分 SKU + 时序感知"

仿真细节：
  · 起点 = 训练集最后一天的 on_hand（来自原始数据）
  · 历史在途订单（test 期间的 arrivals_units）按真实日期到货
  · 三个策略起点完全相同，对比公平
""")


def simulate(actuals: np.ndarray, hist_arrivals: np.ndarray,
             init_on_hand: float, lead_time: int, decision_fn):
    """逐日仿真。

    Args:
        actuals:        长度 28 的当日实际需求
        hist_arrivals:  长度 28 的当日"历史在途订单"到货量（来自原数据 arrivals_units）
        init_on_hand:   仿真起点的在手库存
        lead_time:      新下单的提前期（天）
        decision_fn(day, on_hand, pending_qty) → 当日下单量（0 表示不下）

    每日时序：
      1) 历史在途订单 + 本策略下的新订单 入库
      2) 当日需求消耗库存（无货 → 缺货）
      3) 评审日决定是否下单（lead_time 天后到货）
    """
    on_hand = float(init_on_hand)
    pending: list[tuple[int, float]] = []   # [(arrive_day, qty)]
    stockout_days = 0
    total_demand = 0.0
    served_demand = 0.0
    for day, dmd in enumerate(actuals):
        # 1) 到货：历史在途 + 仿真期下单
        on_hand += float(hist_arrivals[day])
        arrived_qty = sum(q for d, q in pending if d == day)
        pending = [(d, q) for d, q in pending if d > day]
        on_hand += arrived_qty
        # 2) 需求消耗
        served = min(on_hand, float(dmd))
        on_hand -= served
        total_demand += float(dmd)
        served_demand += served
        if served < float(dmd):
            stockout_days += 1
        # 3) 评审决策
        pending_qty = sum(q for _, q in pending)
        order_qty = decision_fn(day, on_hand, pending_qty)
        if order_qty > 0:
            pending.append((day + lead_time, order_qty))
    fill_rate = served_demand / total_demand if total_demand > 0 else 1.0
    return stockout_days, fill_rate


rows_A, rows_B, rows_C = [], [], []
for (sku, wh), pack in forecast_storage.items():
    s_df = (df[(df.sku_id == sku) & (df.warehouse == wh)]
            .sort_values("date").reset_index(drop=True))
    snap_idx = len(s_df) - TEST_HORIZON - 1
    snap = s_df.iloc[snap_idx]
    L  = int(snap["lead_time_days"])
    SS = float(snap["safety_stock_units"])
    init_oh = float(snap["on_hand_units"])
    avg_daily = float(s_df.iloc[:snap_idx + 1]["demand_units"].mean())
    # 测试期里数据中真实发生的到货量（原供应链已下、test 期内陆续到达）
    hist_arrivals = s_df.iloc[snap_idx + 1: snap_idx + 1 + TEST_HORIZON]["arrivals_units"].values

    # 策略 A：全局固定（不分 SKU）
    def decision_A(day, oh, pend, _T=GLOBAL_THRESHOLD, _Q=GLOBAL_ORDER_QTY, _R=REVIEW_PERIOD):
        if day % _R != 0:
            return 0.0
        return _Q if (oh + pend) < _T else 0.0

    # 策略 B：每 SKU 历史均值（静态）
    T_B = avg_daily * L
    Q_B = avg_daily * (L + REVIEW_PERIOD)

    def decision_B(day, oh, pend, _T=T_B, _Q=Q_B, _R=REVIEW_PERIOD):
        if day % _R != 0:
            return 0.0
        return _Q if (oh + pend) < _T else 0.0

    # 策略 C：Prophet 动态 ROP
    yp = pack["y_pred"]

    def decision_C(day, oh, pend, _L=L, _SS=SS, _R=REVIEW_PERIOD, _fc=yp):
        if day % _R != 0:
            return 0.0
        lead_win   = _fc[day:day + _L].sum() if day < len(_fc) else _fc[-_L:].sum()
        review_win = _fc[day:day + _R].sum() if day < len(_fc) else _fc[-_R:].sum()
        rop = lead_win + _SS
        S_  = rop + review_win
        ipos = oh + pend
        return max(0.0, S_ - ipos) if ipos < rop else 0.0

    actuals = pack["y_true"]
    sd_A, fr_A = simulate(actuals, hist_arrivals, init_oh, L, decision_A)
    sd_B, fr_B = simulate(actuals, hist_arrivals, init_oh, L, decision_B)
    sd_C, fr_C = simulate(actuals, hist_arrivals, init_oh, L, decision_C)
    rows_A.append(dict(sku=sku, warehouse=wh, stockout_days=sd_A, fill_rate=fr_A))
    rows_B.append(dict(sku=sku, warehouse=wh, stockout_days=sd_B, fill_rate=fr_B))
    rows_C.append(dict(sku=sku, warehouse=wh, stockout_days=sd_C, fill_rate=fr_C))

df_A = pd.DataFrame(rows_A)
df_B = pd.DataFrame(rows_B)
df_C = pd.DataFrame(rows_C)


def _print_strategy(name, d):
    print(f"\n[{name}]")
    print(f"   平均缺货天数 / SKU·仓 = {d['stockout_days'].mean():.2f} 天 / 28 天")
    print(f"   总缺货天数            = {int(d['stockout_days'].sum())} 天")
    print(f"   平均 Fill Rate        = {d['fill_rate'].mean():.2%}")


_print_strategy("策略 A · 全局固定阈值", df_A)
_print_strategy("策略 B · 每 SKU 历史均值", df_B)
_print_strategy("策略 C · Prophet 动态 ROP", df_C)

# 主对比：C vs A（量化 AI 方案 vs 传统经验式的核心收益）
denom_A = max(int(df_A["stockout_days"].sum()), 1)
denom_B = max(int(df_B["stockout_days"].sum()), 1)
reduction_C_vs_A = (df_A["stockout_days"].sum() - df_C["stockout_days"].sum()) / denom_A
reduction_C_vs_B = (df_B["stockout_days"].sum() - df_C["stockout_days"].sum()) / denom_B
fill_lift_C_vs_A = df_C["fill_rate"].mean() - df_A["fill_rate"].mean()
fill_lift_C_vs_B = df_C["fill_rate"].mean() - df_B["fill_rate"].mean()

print("\n--- 核心对比 ---")
print(f">>> 策略 C vs A（全局固定）  缺货风险相对降低 = {reduction_C_vs_A:.2%}    ← ★ 主指标")
print(f">>> 策略 C vs A             Fill Rate 提升   = {fill_lift_C_vs_A:+.2%}")
print("\n--- 次对比（讲故事时备用：'即使分 SKU 管理，时序感知仍有增益'）---")
print(f">>> 策略 C vs B（每SKU均值） 缺货风险相对降低 = {reduction_C_vs_B:.2%}")
print(f">>> 策略 C vs B             Fill Rate 提升   = {fill_lift_C_vs_B:+.2%}")

# 合并明细表存盘
cmp = (df_A.rename(columns={"stockout_days": "sd_A_global_fixed", "fill_rate": "fr_A"})
           .merge(df_B.rename(columns={"stockout_days": "sd_B_per_sku_mean", "fill_rate": "fr_B"}),
                  on=["sku", "warehouse"])
           .merge(df_C.rename(columns={"stockout_days": "sd_C_prophet", "fill_rate": "fr_C"}),
                  on=["sku", "warehouse"]))
cmp["reduction_C_vs_A"] = cmp["sd_A_global_fixed"] - cmp["sd_C_prophet"]
cmp.to_csv(f"{OUTPUT_DIR}/strategy_comparison.csv", index=False)
print(f"\n[已保存] {OUTPUT_DIR}/strategy_comparison.csv")

# 三策略缺货天数对比柱状图
fig, ax = plt.subplots(figsize=(13, 4.5))
labels = (cmp["sku"] + "/" + cmp["warehouse"]).values
x = np.arange(len(labels))
w = 0.27
ax.bar(x - w, cmp["sd_A_global_fixed"], width=w, color="#d62728", label="A · 全局固定")
ax.bar(x,     cmp["sd_B_per_sku_mean"], width=w, color="#888888", label="B · 每SKU均值")
ax.bar(x + w, cmp["sd_C_prophet"],      width=w, color="#2c7fb8", label="C · Prophet 动态")
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=80, fontsize=7)
ax.set_ylabel("缺货天数 / 28 天")
ax.set_title("三种补货策略的缺货天数对比")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/strategy_comparison.png", dpi=120)
plt.close()
print(f"[已保存] {OUTPUT_DIR}/strategy_comparison.png")

# 兼容下文 SECTION 6 旧变量名
reduction = reduction_C_vs_A
fill_lift = fill_lift_C_vs_A
base_df = df_A
dyn_df  = df_C


# ============ SECTION 6 · 输出汇总 ============
banner("SECTION 6 · 输出文件清单 + 关键 KPI 汇总")

headline = {
    "validation_mae_avg":          round(float(overall_mae), 3),
    "validation_mape_avg":         round(float(overall_mape), 4),
    "validation_relative_mae_avg": round(float(rel_mae), 4),
    "improvement_vs_naive":        round(float(gain), 4),
    "stockout_reduction_pct":      round(float(reduction), 4),
    "fill_rate_lift_pct":          round(float(fill_lift), 4),
    "fill_rate_fixed":             round(float(base_df["fill_rate"].mean()), 4),
    "fill_rate_dynamic":           round(float(dyn_df["fill_rate"].mean()), 4),
}
print("Headline 指标：")
for k, v in headline.items():
    tag = "%" if any(s in k for s in ["mape", "pct", "relative", "rate", "improvement"]) else ""
    if tag == "%":
        print(f"   {k:32s} = {v*100:7.2f} %")
    else:
        print(f"   {k:32s} = {v:7.3f} units")

pd.DataFrame([headline]).to_csv(f"{OUTPUT_DIR}/headline_metrics.csv", index=False)

print(f"\n[文件清单] outputs/")
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f"   - {f}")

print("""\n
============================================================================
 实验完成 ✓
 关键 KPI ①：MAE < {:.1f}%（字段 validation_relative_mae_avg）
 关键 KPI ②：缺货风险降低约 {:.1f}%（字段 stockout_reduction_pct）
 交付形态：Python 实验脚本 + Streamlit 看板（运行 02_streamlit_app.py 启动看板）
============================================================================
""".format(rel_mae * 100, reduction * 100))
