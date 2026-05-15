# 智能备件需求预测与补货决策工具
> 基于 20 SKU × 2 仓库 × 10,000+ 条日级历史消耗数据，用 **Prophet** 预测未来 4 周需求，结合 **(R, s, S) 动态再订货点策略** 替代传统经验式补货——在测试期 28 天仿真中，**缺货风险降低约 65%**，Fill Rate 由 **94.9% → 98.3%**。

端到端 demo：Python 实验脚本 + Streamlit 交互看板。一条 `run.bat` 启动全部。

---

## 1. 核心结果

### 1.1 关键指标

| 指标 | 数值 | 出处文件 | 出处字段 |
|---|---|---|---|
| 全 20 SKU 验证集 **相对 MAE** | **29.98%** | `sample_outputs/headline_metrics.csv` | `validation_relative_mae_avg` |
| 全 20 SKU 验证集 sMAPE | 31.75% | `sample_outputs/validation_metrics_per_sku.csv` | `smape` (mean) |
| Prophet 相对 naive 均值 MAE 改善 | +5.34% | `sample_outputs/headline_metrics.csv` | `improvement_vs_naive` |
| **缺货风险（缺货天数）相对降低** | **64.94%** | `sample_outputs/headline_metrics.csv` | `stockout_reduction_pct` |
| Fill Rate 提升 | +3.46 pp | `sample_outputs/headline_metrics.csv` | `fill_rate_lift_pct` |
| 基线策略 A（全局固定）Fill Rate | 94.88% | `sample_outputs/headline_metrics.csv` | `fill_rate_fixed` |
| 主策略 C（Prophet 动态）Fill Rate | 98.33% | `sample_outputs/headline_metrics.csv` | `fill_rate_dynamic` |

### 1.2 系统界面

#### Streamlit 看板

**主看板** — Prophet 预测曲线 + 关键指标卡 + 补货建议
![dashboard](screenshot_dashboard.png)

**AI 助手 Tab 入口** — 示例问题 + 对话输入框 + 智谱 API Key 配置
![agent tab](screenshot_agent_tab.png)

**Agent 工具调用测试** — 自然语言提问 → 自动调用 `compute_replenishment` → 结构化决策表
![agent demo](screenshot_agent_demo.png)

#### 脚本自动产出的分析图

**EDA 概览**（每日总需求 + 各 SKU 平均需求）
![eda](sample_outputs/eda_overview.png)

**Prophet 4 周预测 vs 实际**（示例：SKU001 @ WH_A）
![demo forecast](sample_outputs/demo_forecast_SKU001_WH_A.png)

**三种补货策略缺货天数对比**（28 天测试期）
![strategy comparison](sample_outputs/strategy_comparison.png)

---

## 2. 数据集

- **来源**：Kaggle —— Inventory Replenishment Timeseries（共 10,000 条记录）
- **规模**：20 SKU × 2 仓库（WH_A / WH_B）= 40 条序列，每条约 250 个连续日
- **时间范围**：2025-05-11 → 2026-01-15（约 8 个月）
- **关键字段**：
  - `demand_units`（预测目标） · `demand_forecast_units`（数据自带的基线预测）
  - `lead_time_days` · `safety_stock_units` · `service_level_target`
  - `on_hand_units` · `on_order_units` · `arrivals_units` · `order_qty_units`
  - `holiday_flag` · `promo_flag` · `weather_index` · `price_usd`
  - `stockout`（0/1） · `fill_rate` · 各类成本列

---

## 3. 项目结构

```
Supply_Chain_AI_Project/
├── README.md
├── run.bat                                       # Windows 一键运行
├── run.sh                                        # macOS / Linux 一键运行
├── requirements.txt                              # pip 依赖
├── environment.yml                               # conda 环境（推荐 Windows 用户）
├── .gitignore
│
├── 01_demand_forecast_and_replenishment.py       # 主实验脚本（6 SECTION）
├── 02_streamlit_app.py                           # Streamlit 交互看板（含 AI 助手 Tab）
├── agent_tools.py                                # 5 个工具函数 + JSON Schema + dispatch
├── agent_core.py                                 # GLM tool-use 手动 loop（ZhipuAI SDK）
│
├── inventory_replenishment_timeseries_10000.csv  # 原始数据
│
├── sample_outputs/                               # 演示快照（已 commit）
│   ├── headline_metrics.csv                      #   ← 核心 KPI 唯一出处
│   ├── validation_metrics_per_sku.csv            #     每 SKU 的预测指标
│   ├── replenishment_recommendation.csv          #     当前补货建议表
│   ├── strategy_comparison.csv                   #     三策略缺货天数对比
│   ├── sku_warehouse_summary.csv                 #     每 SKU/仓基础统计
│   ├── eda_overview.png
│   ├── demo_forecast_SKU001_WH_A.png
│   └── strategy_comparison.png
│
├── outputs/        # 每次跑实验脚本的实时产物（.gitignore）
└── run_logs/       # 每次运行的完整 stdout/stderr 日志（.gitignore）
                    # 命名格式：YYYYMMDDHHMM.txt
```

---

## 4. 快速开始

### 4.1 启动系统

#### 方式 A · 一键脚本

**Windows**：双击 `run.bat`，或在终端里
```bat
.\run.bat
```

**macOS / Linux**：
```bash
chmod +x run.sh
./run.sh
```

脚本会依次执行：安装依赖 → 跑实验脚本 → 启动 Streamlit 看板。

#### 方式 B · conda 环境

```bash
conda env create -f environment.yml
conda activate supplychain
python 01_demand_forecast_and_replenishment.py
streamlit run 02_streamlit_app.py
```

#### 方式 C · pip

```bash
pip install -r requirements.txt
python 01_demand_forecast_and_replenishment.py
streamlit run 02_streamlit_app.py
```

> **重要：看板必须用 `streamlit run`，不能用 `python`！**
> 用 `python 02_streamlit_app.py` 会刷一堆 `missing ScriptRunContext` 警告且看板不会出现——这是 Streamlit 必须由 `streamlit` CLI 启动才能拉起内部服务。

启动看板后浏览器自动打开 `http://localhost:8501`，按 `Ctrl+C` 退出。

### 4.2 AI 助手 Tab —— 需要智谱 AI API Key

主看板（Prophet 预测 + 补货建议）不需要 Key 即可使用。**AI 助手 Tab 调用智谱 GLM**，需要：

1. 注册并申请 Key：https://open.bigmodel.cn/usercenter/apikeys
2. 两种用法（任选其一）：
   - **临时**：在 Streamlit 侧栏 "智谱 AI API Key" 框粘贴
   - **永久**（Windows PowerShell）：
     ```powershell
     [Environment]::SetEnvironmentVariable("ZHIPUAI_API_KEY", "你的key", "User")
     ```
3. 默认模型 `glm-4-plus`；如需省钱测试，改 `agent_core.py` 中的 `MODEL = "glm-4-flash"`

⚠️ **不要把 Key 写入代码或 commit 到 git**。本项目的 `.gitignore` 已排除 `.env` / `secrets.toml` 等典型密钥文件。

---

## 5. 方法论

### 5.1 预测：Prophet + 节假日/促销外生回归

- 每 SKU/仓单独建模（40 条时间序列）
- 训练 222 天，验证 28 天（留出法）
- 启用 `weekly_seasonality`，禁用 `yearly_seasonality`（数据只有 8 个月）
- 加入 `holiday_flag`、`promo_flag` 作为额外回归变量
- `interval_width=0.95` → 95% 预测区间

### 5.2 补货：(R, s, S) 周期评审 + 动态 ROP

每个评审日 (R = 7) 决策：
- **再订货点** ROP = Prophet 预测未来 L 天需求 + 安全库存
- **订到点** S = ROP + Prophet 预测未来 R 天需求
- 当 *库存位置* (on_hand + on_order) < ROP 时，下单到 S

### 5.3 策略对比（28 天滚动仿真）

三策略起点相同，历史在途订单（`arrivals_units`）按真实日期到货：

| 策略 | 说明 | 缺货天数/SKU·仓 | Fill Rate |
|---|---|---|---|
| A · 全局固定阈值 | 全 SKU 共用 T=100, Q=200（"管理员拍脑袋"基线） | 1.93 | 94.88% |
| B · 每 SKU 历史均值 | ROP=avg×L，分 SKU 但静态 | 0.68 | 98.33% |
| **C · Prophet 动态 ROP** | 分 SKU + 时序感知（主策略） | **0.68** | **98.33%** |

### 5.4 AI 助手层：智谱 GLM + tool-use

`agent_tools.py` 把上述 Prophet/ROP 计算封装为 5 个工具函数，Agent 通过 ZhipuAI SDK（OpenAI 兼容协议）动态调用：

| 工具 | 作用 |
|---|---|
| `list_skus` | 列出所有可用 SKU/仓库组合 |
| `get_inventory_status` | 查 SKU 当前库存、安全库存、提前期、历史 Fill Rate |
| `forecast_demand` | 调 Prophet 预测未来 N 天需求，附验证集 MAE |
| `compute_replenishment` | 计算动态 ROP + 订到点 S + 建议下单量 |
| `place_order` | **沙盒**模拟下单（写日志到 `outputs/sandbox_orders.log`，不触发真实采购） |

[agent_core.py](agent_core.py) 实现的 tool-use 手动循环参考 OpenAI / ZhipuAI 标准模式：`finish_reason == "tool_calls"` 时执行所有工具 → `role:"tool"` 消息塞回 → 继续直到 `finish_reason == "stop"`。

---

## 6. 核心 KPI 出处对照表

| 指标 | 数值 | 字段 | 出处 |
|---|---|---|---|
| 验证集相对 MAE（20 SKU × 4 周预测） | 29.98% | `validation_relative_mae_avg` | `sample_outputs/headline_metrics.csv` |
| 缺货风险相对降低（动态 ROP vs 全局固定基线） | 64.94% | `stockout_reduction_pct` | `sample_outputs/headline_metrics.csv` |
| Fill Rate 提升 | 94.88% → 98.33%（+3.46 pp） | `fill_rate_fixed` / `fill_rate_dynamic` | `sample_outputs/headline_metrics.csv` |
| 交付形态 | Python 脚本 + Streamlit 看板 | `01_*.py` / `02_*.py` + `sample_outputs/*.png` | 本仓库 |

---

## 7. 已知局限与设计取舍

诚实陈述项目当前的边界与权衡：

1. **该数据需求平稳**，Prophet 相对 naive 均值改善有限（+5.34% MAE）。Section 5 中策略 B 与 C 持平也印证这点。**真实部署建议做 model selection**——按 sMAPE 在 naive / Prophet / Croston-SBA 之间逐 SKU 择优。
2. **备件需求常为间歇性**（很多天 0 需求），MAPE 在此场景数学上必然爆炸——已用 sMAPE / 相对 MAE 替代。
3. **未对外生变量做未来值预测**（如 `weather_index`），目前只使用了已知的 `holiday_flag` / `promo_flag`。
4. **滚动验证未做** —— 仅单次留出 28 天。生产建议改 expanding-window CV。

---

## 8. 路线图

- [x] **Phase 1 · MVP**：Prophet + 动态 ROP + Streamlit 看板（当前）
- [ ] **Phase 2 · Model Zoo**：补充 Croston-SBA / SARIMAX / LightGBM，每 SKU 选最优
- [ ] **Phase 3 · 成本建模**：把 holding / stockout / ordering 三栏 cost 纳入目标函数，做总成本优化
- [x] **Phase 4 · 对话 Agent**：基于智谱 GLM（OpenAI 兼容协议）的 tool-use，自然语言提问 → 工具调用 → 决策建议 + 沙盒下单

---

## 9. 复现说明

要复现 `sample_outputs/` 里的数字：

```bash
python 01_demand_forecast_and_replenishment.py
```

新产物落到 `outputs/`，**不覆盖** `sample_outputs/`。完整 stdout 日志会写到 `run_logs/YYYYMMDDHHMM.txt`。

随机性来源：Prophet 内部 MCMC（实际用 MAP 估计时为零）。同机环境多次复跑结果应位级一致。

---

## 10. License

MIT License — 见 [LICENSE](LICENSE)

## 11. 联系方式

如有问题，请直接开 issue。

> 数据来自 Kaggle 公开数据集；所有指标可基于仓库代码完整复现。
