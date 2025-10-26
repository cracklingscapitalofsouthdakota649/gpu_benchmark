#!/usr/bin/env bash
set -euo pipefail

# Allow pytest to fail while we still validate Allure output presence
set +e

OS_NAME="${1:-ubuntu-latest}"
RESULTS_DIR="allure-results/$OS_NAME"
mkdir -p "$RESULTS_DIR"

if [[ "$OS_NAME" == "ubuntu-latest" ]]; then
  PATH_SEP=":"
  echo "🐧 Running tests on Ubuntu (Direct Runner)..."
else
  PATH_SEP=";"
  echo "🦾 Running tests on Windows..."
fi

cd "${GITHUB_WORKSPACE:-$PWD}"
export PYTHONPATH="${GITHUB_WORKSPACE:-$PWD}${PATH_SEP}${PYTHONPATH:-}"

# ---------- Pre-flight diagnostics (import visibility) ----------
python .github/scripts/preflight.py

# Run pytest with importlib mode
python -m pytest -v -m gpu --alluredir="$RESULTS_DIR" --import-mode=importlib
TEST_EXIT_CODE=$?

# Verify Allure results exist (setup/collection succeeded)
if [ ! -d "$RESULTS_DIR" ] || [ -z "$(find "$RESULTS_DIR" -type f -name '*.json' 2>/dev/null)" ]; then
  echo "❌ No Allure results found — setup or configuration failure."
  echo "test_outcome=FAIL" >> "$GITHUB_OUTPUT"
  exit 1
fi

if [ $TEST_EXIT_CODE -ne 0 ]; then
  echo "⚠️ Some tests failed (pytest exit code $TEST_EXIT_CODE)."
  echo "test_outcome=UNSTABLE" >> "$GITHUB_OUTPUT"
  exit 0
else
  echo "✅ All tests passed successfully."
  echo "test_outcome=PASS" >> "$GITHUB_OUTPUT"
  exit 0
fi