@echo off
:: run.bat — Quick launcher for SPY Quant on Windows
:: Double-click or run from cmd:  run.bat [command]
::
:: Commands:
::   run.bat train       — Train the model (Alpaca data)
::   run.bat evaluate    — Backtest / evaluate
::   run.bat optimize    — Walk-forward optimization
::   run.bat live        — Start paper trading loop
::   run.bat dashboard   — Start web dashboard (http://localhost:8080)
::   run.bat monitor     — Terminal status dashboard
::   run.bat report      — Generate today's HTML report
::   run.bat shell       — Activate venv in a new shell

setlocal
set "ROOT=%~dp0"
set "PYTHON=%ROOT%venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo.
    echo  ERROR: venv not found. Run setup_windows.ps1 first.
    echo.
    pause
    exit /b 1
)

set CMD=%1
if "%CMD%"=="" set CMD=help

if "%CMD%"=="train"     goto :train
if "%CMD%"=="evaluate"  goto :evaluate
if "%CMD%"=="optimize"  goto :optimize
if "%CMD%"=="live"      goto :live
if "%CMD%"=="dashboard" goto :dashboard
if "%CMD%"=="monitor"   goto :monitor
if "%CMD%"=="report"    goto :report
if "%CMD%"=="shell"     goto :shell
goto :help

:train
echo Starting training (Alpaca data)...
"%PYTHON%" scripts\train.py --alpaca
goto :end

:evaluate
echo Running evaluation...
"%PYTHON%" scripts\evaluate.py --alpaca
goto :end

:optimize
echo Running walk-forward optimization...
"%PYTHON%" scripts\optimize.py --alpaca
goto :end

:live
echo Starting paper trading loop...
echo Press Ctrl+C to stop.
"%PYTHON%" scripts\run_live.py
goto :end

:dashboard
echo Starting dashboard at http://localhost:8080
start "" "http://localhost:8080"
"%PYTHON%" -m dashboard.server
goto :end

:monitor
"%PYTHON%" scripts\monitor.py --watch
goto :end

:report
"%PYTHON%" scripts\report.py
goto :end

:shell
echo Activating virtual environment...
call "%ROOT%venv\Scripts\activate.bat"
cmd /k
goto :end

:help
echo.
echo  SPY Quant Launcher
echo  ------------------
echo  run.bat train       Train model (pulls Alpaca data)
echo  run.bat evaluate    Backtest ^& evaluate
echo  run.bat optimize    Walk-forward optimization
echo  run.bat live        Paper trading loop
echo  run.bat dashboard   Web dashboard (port 8080)
echo  run.bat monitor     Terminal status dashboard
echo  run.bat report      Generate HTML session report
echo  run.bat shell       Open venv shell
echo.

:end
endlocal
