@echo off
REM ============================================================================
REM Round two. Times every ablation value against an unmodified reference.
REM
REM Round one proved the code paths work. It also showed that the recurrent
REM update, not the Unity simulation, is what costs the time on level 5, which
REM makes the long sequences and the burn-in runs a walltime risk on the
REM cluster. These five runs supply the numbers to decide that before any
REM 48-hour job is submitted.
REM
REM 10 is the unmodified production config and is the denominator for every
REM ratio. Without it the other four timings mean nothing, so run all five.
REM
REM Activate the environment first, then run this from anywhere:
REM     conda activate drl_nav_hpc
REM     D:\Projects\temporal-sac-dynamic-navigation\run_smoke_tests_round2.bat
REM
REM Logs go to the same smoke_logs folder as round one, numbered 10 to 14.
REM Expect roughly 30 minutes in total, based on the round one timings.
REM ============================================================================

setlocal
set ROOT=D:\Projects\temporal-sac-dynamic-navigation
set LOGS=%ROOT%\smoke_logs

REM Same folder as round one. Change this one line if the configs move.
set CONFIGDIR=ablation/smoke

if not exist "%LOGS%" mkdir "%LOGS%"

REM RunPolicy resolves the build and config folders relative to the working
REM directory, so it has to be launched from inside Algorithms.
cd /d "%ROOT%\Algorithms"

REM env_id values continue the round one sequence, still even and spaced by two.

REM --- the reference, which every ratio is measured against ----------------
call :run 10_ref sac_lstm_abl_reference_depthwise_SMOKE.yaml 88

REM --- the four ablation values that round one did not time ---------------
call :run 11_seq05 sac_lstm_abl_seq05_depthwise_SMOKE.yaml 90
call :run 12_seq15 sac_lstm_abl_seq15_depthwise_SMOKE.yaml 92
call :run 13_mem256 sac_lstm_abl_mem256_depthwise_SMOKE.yaml 94
call :run 14_burnin05 sac_lstm_abl_burnin05_depthwise_SMOKE.yaml 96

echo.
echo ============================================================================
echo All five runs attempted. Now check and time them:
echo     cd /d %ROOT%
echo     python results_check\check_smoke_logs.py smoke_logs
echo     python results_check\measure_throughput.py smoke_logs
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
