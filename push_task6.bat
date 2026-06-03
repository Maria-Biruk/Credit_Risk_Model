@echo off
REM Push local task-6 branch to origin and set upstream

nREM Ensure we're in the repo root before running this script.
cd /d %~dp0

n:: Try to checkout task-6 (will succeed if already on it)
git checkout task-6 || (
  echo Failed to checkout branch 'task-6'. Ensure it exists locally.
  exit /b 1
)

n:: Push and set upstream
git push -u origin task-6 || (
  echo git push failed. Check your network/auth and try again.
  exit /b 1
)

necho Successfully pushed 'task-6' to origin.
exit /b 0
