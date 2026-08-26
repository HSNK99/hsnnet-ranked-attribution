@echo off
set PYTHONDONTWRITEBYTECODE=1
python scripts\run_all_tests.py
if errorlevel 1 exit /b 1
echo HSNNET TESTS PASSED.
