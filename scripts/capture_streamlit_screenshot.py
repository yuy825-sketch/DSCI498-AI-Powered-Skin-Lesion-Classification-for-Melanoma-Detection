from __future__ import annotations

import argparse
import base64
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


def _http_json(method: str, url: str, payload: dict | None = None, timeout: float = 10.0) -> dict:
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    return json.loads(body.decode("utf-8"))


def _wait_url(url: str, timeout_s: float = 60.0) -> None:
    start = time.time()
    last_err = None
    while time.time() - start < timeout_s:
        try:
            urllib.request.urlopen(url, timeout=1).read(200)
            return
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(0.5)
    raise RuntimeError(f"Streamlit not ready at {url!r}: {last_err}")


def _wait_rendered(base_url: str, session_id: str, *, timeout_s: float = 60.0) -> None:
    start = time.time()
    last = ""
    while time.time() - start < timeout_s:
        try:
            res = _http_json(
                "POST",
                f"{base_url}/session/{session_id}/execute/sync",
                {
                    "script": (
                        "return {\n"
                        "  readyState: document.readyState,\n"
                        "  textLen: (document.body && document.body.innerText) ? document.body.innerText.length : 0,\n"
                        "  hasH1: !!document.querySelector('h1'),\n"
                        "  h1: document.querySelector('h1') ? document.querySelector('h1').innerText : '',\n"
                        "};"
                    ),
                    "args": [],
                },
                timeout=5.0,
            )
            v = res.get("value", {})
            text_len = int(v.get("textLen", 0) or 0)
            h1 = str(v.get("h1", "") or "")
            last = f"readyState={v.get('readyState')} textLen={text_len} h1={h1!r}"
            # Streamlit pages render text content asynchronously; wait until it's clearly populated.
            if text_len > 200 and ("Skin Lesion" in h1 or "Skin Lesion" in str(v)):
                return
            if text_len > 400:
                return
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.75)
    raise RuntimeError(f"Timed out waiting for Streamlit to render: {last}")


def _start_geckodriver(port: int) -> subprocess.Popen:
    log_path = Path("/tmp/geckodriver_streamlit.log")
    log_f = log_path.open("wb")
    proc = subprocess.Popen(
        ["geckodriver", "--host", "127.0.0.1", "--port", str(port), "--log", "info"],
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return proc


def _wait_geckodriver(base_url: str, timeout_s: float = 10.0) -> None:
    start = time.time()
    last_err = None
    while time.time() - start < timeout_s:
        try:
            _http_json("GET", f"{base_url}/status", None, timeout=1.0)
            return
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(0.2)
    raise RuntimeError(f"geckodriver not ready at {base_url}: {last_err}")


def _stop_proc(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:  # noqa: BLE001
        proc.terminate()
    try:
        proc.wait(timeout=5)
    except Exception:  # noqa: BLE001
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:  # noqa: BLE001
            proc.kill()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="http://localhost:8502/?demo=1")
    parser.add_argument("--out", type=Path, default=Path("results/streamlit_demo.png"))
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--port", type=int, default=4444)
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args()

    _wait_url(args.url, timeout_s=min(args.timeout, 60.0))

    driver = _start_geckodriver(args.port)
    base = f"http://127.0.0.1:{args.port}"
    try:
        _wait_geckodriver(base, timeout_s=10.0)
        # Create session
        caps = {
            "capabilities": {
                "alwaysMatch": {
                    "browserName": "firefox",
                    "acceptInsecureCerts": True,
                    "moz:firefoxOptions": {"args": ["-headless"]},
                }
            }
        }
        res = _http_json("POST", f"{base}/session", caps, timeout=15.0)
        session_id = res["value"]["sessionId"]

        # Window size
        _http_json(
            "POST",
            f"{base}/session/{session_id}/window/rect",
            {"width": int(args.width), "height": int(args.height)},
            timeout=10.0,
        )

        # Navigate
        _http_json("POST", f"{base}/session/{session_id}/url", {"url": args.url}, timeout=10.0)

        # Wait for Streamlit to render real content
        _wait_rendered(base, session_id, timeout_s=float(args.timeout))

        # Screenshot
        shot = _http_json("GET", f"{base}/session/{session_id}/screenshot", None, timeout=20.0)
        b64 = shot["value"]
        png = base64.b64decode(b64)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(png)
        print("Wrote:", args.out)

        # Delete session
        try:
            _http_json("DELETE", f"{base}/session/{session_id}", None, timeout=10.0)
        except Exception:
            pass
    finally:
        _stop_proc(driver)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
