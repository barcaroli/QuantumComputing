@echo off
REM ============================================================
REM  EXPERIMENT: frame 20 atomic strata
REM  Y = Airbat, Surfacesbois   |   CV = 0.05   |   K = 4
REM ============================================================
REM  All methods on the same input (swiss_municipalities.csv),
REM  with pre-categorized target variables: POPTOT.cat / HApoly.cat.
REM ============================================================
REM De-comment the line you want to execute
REM and indicate the token if required

set CSV=swiss_municipalities.csv
set COMMON=--csv %CSV% --cv 0.05 --y1 Airbat --y2 Surfacesbois --benchmark 129

REM --- D-Wave (simulate, best-of-10) ---
REM python 2.DWave_quantum_stratification_qpu.py --mode simulate %COMMON% --K 4 --num-reads 300 --num-sweeps 20000 --best-of 10 > out20_dwave_K4.txt 2>&1

REM --- IBM K=4 on real hardware ---
REM NOTE: 3.IBM_quantum_stratification.py does not accept --cv/--y1/--y2/--benchmark
REM (Y1/Y2 are hardcoded to Surfacesbois/Airind), so %COMMON% is NOT used here.
REM python -X utf8 3.IBM_quantum_stratification.py --mode hardware --csv %CSV% --K 4 --n-macro 20 --p 1 --maxiter 2 --token "IBM_token" > out20_ibm_K4_hw.txt 2>&1

REM --- Dirac-3 QUBO one-hot ---
REM python 4.DIRAC3_quantum_stratification_qci.py %COMMON% --K 4 --formulation qubo --token "Dirac-3_token" > out20_dirac_K4.txt 2>&1
REM  Test offline:
REM python 4.DIRAC3_quantum_stratification_qci.py %COMMON% --K 4 --formulation qubo --dry-run > out20_dirac_dryrun.txt 2>&1

echo Experiment 20 strata completed. Output in out20_*.txt
pause
