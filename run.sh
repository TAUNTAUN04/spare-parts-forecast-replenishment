#!/usr/bin/env bash
# ===========================================================================
#  一键运行（macOS / Linux）：安装依赖 + 跑实验 + 启动 Streamlit 看板
#  用法：chmod +x run.sh && ./run.sh
#  覆盖 Python 解释器：PYTHON=/path/to/python ./run.sh
# ===========================================================================
set -euo pipefail

PYTHON="${PYTHON:-python3}"

if ! command -v "$PYTHON" > /dev/null 2>&1; then
    echo "[ERROR] 找不到 $PYTHON，请安装 Python 3.9+ 或设置 PYTHON 环境变量" >&2
    exit 1
fi

echo "Using Python: $($PYTHON --version)"
echo

echo "====== [1/3] 安装依赖 ======"
"$PYTHON" -m pip install -r requirements.txt

echo
echo "====== [2/3] 运行实验脚本 ======"
"$PYTHON" 01_demand_forecast_and_replenishment.py

echo
echo "====== [3/3] 启动 Streamlit 看板（Ctrl+C 退出） ======"
echo "浏览器会自动打开 http://localhost:8501"
"$PYTHON" -m streamlit run 02_streamlit_app.py
