@echo off
REM Wrapper to run flake8 via the Python launcher (ensures command works on Windows CMD)
py -3.14 -m flake8 %*
