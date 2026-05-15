@echo off
REM ===========================================================================
REM  一键运行：安装依赖 + 跑实验 + 启动 Streamlit 看板
REM  用法：双击 run.bat，或在 cmd / PowerShell 里执行  .\run.bat
REM ===========================================================================
chcp 65001 > nul
setlocal

REM --- 自动定位 Python：PATH > py launcher > 常见 miniconda 路径 ---
set PYTHON=
where python > nul 2>&1
if not errorlevel 1 set PYTHON=python

if "%PYTHON%"=="" (
    where py > nul 2>&1
    if not errorlevel 1 set PYTHON=py
)

if "%PYTHON%"=="" (
    if exist "D:\miniconda3new\python.exe" set PYTHON=D:\miniconda3new\python.exe
)

if "%PYTHON%"=="" (
    echo [ERROR] 找不到 Python。请先安装 Python 3.9+ 或激活你的 conda 环境再试。
    pause
    exit /b 1
)

echo Using Python: %PYTHON%
%PYTHON% --version
echo.

echo ====== [1/3] 安装依赖 ======
%PYTHON% -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERROR] 依赖安装失败。常见解决：
    echo   * 若 Prophet/cmdstanpy 报权限错误，重开 cmd 用管理员权限再跑
    echo   * 或先  conda create -n supplychain python=3.11  再  conda activate supplychain
    pause
    exit /b 1
)

echo.
echo ====== [2/3] 运行实验脚本（约 30~90 秒，输出到 outputs/ 与 run_logs/）======
%PYTHON% 01_demand_forecast_and_replenishment.py
if errorlevel 1 (
    echo [ERROR] 实验脚本运行失败，请查看 run_logs/ 下最新日志。
    pause
    exit /b 1
)

echo.
echo ====== [3/3] 启动 Streamlit 看板 ======
echo 浏览器会自动打开 http://localhost:8501
echo 在此窗口按 Ctrl+C 退出看板
echo.
%PYTHON% -m streamlit run 02_streamlit_app.py

endlocal
