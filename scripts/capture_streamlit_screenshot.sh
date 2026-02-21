#!/usr/bin/env bash
set -euo pipefail

# Captures a Streamlit demo screenshot using Firefox headless.
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

TMP_SHOT="${HOME}/streamlit_demo.png"
URL="http://localhost:${PORT}/?demo=1"

"${PYTHON_BIN}" -m streamlit run app/app.py \
  --server.headless true \
  --server.port "${PORT}" \
  --server.runOnSave false \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  >/tmp/streamlit_demo.log 2>&1 &
pid=$!

"${PYTHON_BIN}" - <<PY
import time, urllib.request
url = "${URL}"
for _ in range(90):
    try:
        urllib.request.urlopen(url, timeout=1).read(200)
        print("ready")
        break
    except Exception:
        time.sleep(1)
else:
    raise SystemExit("streamlit not ready")
PY

firefox --headless --window-size "${WINDOW_SIZE}" --screenshot "${TMP_SHOT}" "${URL}" >/tmp/firefox_screenshot.log 2>&1

kill "${pid}" >/dev/null 2>&1 || true
sleep 1

"${PYTHON_BIN}" - <<PY
from pathlib import Path
src = Path("${TMP_SHOT}")
dst = Path("${OUT_PATH}")
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_bytes(src.read_bytes())
print("wrote", dst)
PY

ls -la "${OUT_PATH}"

