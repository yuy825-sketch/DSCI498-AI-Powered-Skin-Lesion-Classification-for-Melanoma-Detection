#!/usr/bin/env bash
set -euo pipefail

# Captures a Streamlit demo screenshot using geckodriver (WebDriver) so it can
# wait for Streamlit to render (avoids blank/white screenshots).
#
# Usage example (with the project conda env):
#   export LD_PRELOAD=$CONDA_PREFIX/lib/libittnotify.so
#   export DSCI498_DEMO_RUN_DIR=runs/<your_run_dir>
#   export DSCI498_DEMO_IMAGE_SIZE=260
#   export DSCI498_DEMO_AUTO=1
#   PYTHON_BIN="$CONDA_PREFIX/bin/python" bash scripts/capture_streamlit_screenshot.sh

PYTHON_BIN="${PYTHON_BIN:-python3}"
PORT="${PORT:-8502}"
WINDOW_SIZE="${WINDOW_SIZE:-1400,900}"
OUT_PATH="${OUT_PATH:-results/streamlit_demo.png}"

URL="http://localhost:${PORT}/?demo=1"

"${PYTHON_BIN}" -m streamlit run app/app.py \
  --server.headless true \
  --server.port "${PORT}" \
  --server.runOnSave false \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  >/tmp/streamlit_demo.log 2>&1 &
pid=$!

python3 - <<PY
from pathlib import Path
w,h = "${WINDOW_SIZE}".split(",")
print("capturing", "${URL}", "->", "${OUT_PATH}", "size", w, h)
PY

"${PYTHON_BIN}" scripts/capture_streamlit_screenshot.py \
  --url "${URL}" \
  --out "${OUT_PATH}" \
  --width "${WINDOW_SIZE%,*}" \
  --height "${WINDOW_SIZE#*,}" \
  --timeout 120

kill "${pid}" >/dev/null 2>&1 || true
sleep 1

ls -la "${OUT_PATH}"
