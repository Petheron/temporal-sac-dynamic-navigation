@echo off
REM ============================================================================
REM Runs the nine local smoke tests in order, lowest risk first.
REM Each run is 3000 steps and takes a few minutes.
REM
REM Activate the environment first, then run this from anywhere:
REM     conda activate drl_nav_hpc
REM     D:\Projects\temporal-sac-dynamic-navigation\run_smoke_tests.bat
REM
REM Logs are written to smoke_logs and are what gets sent for assessment.
REM Re-running overwrites the logs, so a repeat run is safe.
REM Results folders are not overwritten, so repeats gain a _0 and _1 suffix.
REM That only affects folder names, never the logs or the verdict.
REM ============================================================================

setlocal
set ROOT=D:\Projects\temporal-sac-dynamic-navigation
set LOGS=%ROOT%\smoke_logs

REM The smoke configs live in a subfolder of Config, kept apart from the
REM production ones. RunPolicy builds its config path as
REM     ../Environments/<env>/Config/<batch_config_dir>/<config>
REM so CONFIGDIR is the part that goes in the middle, passed with -bc.
REM Use forward slashes here, because that string is concatenated into a
REM path that already uses them.
REM Change this one line if the configs move.
set CONFIGDIR=ablation/smoke

if not exist "%LOGS%" mkdir "%LOGS%"

REM RunPolicy resolves the build and config folders relative to the working
REM directory, so it has to be launched from inside Algorithms.
cd /d "%ROOT%\Algorithms"

REM env_id values are even and spaced by two, because OffPolicyMain silently
REM rounds an odd value up and reserves worker_id + 1 for the eval environment.
REM Distinct values also stop a lingering Unity process blocking the next port.

REM --- the dispatch probe, which proves Edit 13.1 --------------------------
call :run 01_bl_dispatch sac_baseline_dispatch_check_SMOKE.yaml 70

REM --- the three ablation paths that exercise new code --------------------
call :run 02_burnin10 sac_lstm_abl_burnin10_depthwise_SMOKE.yaml 72
call :run 03_mem128 sac_lstm_abl_mem128_depthwise_SMOKE.yaml 74
call :run 04_seq20 sac_lstm_abl_seq20_depthwise_SMOKE.yaml 76

REM --- the three static levels, one architecture each ---------------------
call :run 05_lstm_nav sac_lstm_static_navigation_SMOKE.yaml 78
call :run 06_lstm_wind sac_lstm_static_winding_SMOKE.yaml 80
call :run 07_lstm_tower sac_lstm_static_tower_SMOKE.yaml 82

REM --- the two remaining architectures on one static level ----------------
call :run 08_bl_nav sac_baseline_static_navigation_SMOKE.yaml 84
call :run 09_tf_nav sac_transformer_static_navigation_SMOKE.yaml 86

echo.
echo ============================================================================
echo All nine runs attempted. Now check them:
echo     cd /d %ROOT%
echo     python results_check\check_smoke_logs.py smoke_logs
echo ============================================================================
goto :eof

REM %1 is the log name, %2 is the config, %3 is the env_id.
:run
echo.
echo ===== %~1 =====
if not exist "%ROOT%\Environments\DRLNavigation\Config\%CONFIGDIR:/=\%\%~2" (
    echo   SKIPPED, config not found at Config\%CONFIGDIR:/=\%\%~2
    goto :eof
)
python RunPolicy.py -r 1 -c "%~2" -bc "%CONFIGDIR%" -e "DRLNavigation" -t 20 -id %~3 -n "_smk_%~1" --no_graphics > "%LOGS%\%~1.log" 2>&1
REM A non-zero code here means RunPolicy itself fell over.
REM A training failure inside the child process will not show up here.
REM Only check_smoke_logs.py can tell you whether the run did the right thing.
if errorlevel 1 (echo   RunPolicy exited non-zero, see %LOGS%\%~1.log) else (echo   finished, see %LOGS%\%~1.log)
goto :eof
