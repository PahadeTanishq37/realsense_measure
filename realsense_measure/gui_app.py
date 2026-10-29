"""
Web-based GUI for the RealSense 3D Scanner.

Launches a Flask + SocketIO server that provides a rich interface for
configuring scan parameters, running the pipeline, and viewing results
in real-time — all through the browser.

Usage:
    python gui_app.py          # opens http://localhost:5050
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
import glob
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
from flask_socketio import SocketIO, emit

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "gui_templates"
STATIC_DIR = BASE_DIR / "gui_static"

app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR),
    static_folder=str(STATIC_DIR),
    static_url_path="/static",
)
app.config["SECRET_KEY"] = "realsense-scanner-2026"
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_process: subprocess.Popen | None = None
_process_lock = threading.Lock()
_is_running = False
_scan_start_time: float | None = None


def _get_scan_folders() -> list[dict]:
    """Return a list of past scan session folders with metadata."""
    stage_dir = BASE_DIR / "stage_wise_output"
    folders = []
    if stage_dir.exists():
        for p in sorted(stage_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if p.is_dir() and p.name.startswith("scan_test_"):
                meta_file = p / "measurement.json"
                info = {"name": p.name, "path": str(p)}
                if meta_file.exists():
                    try:
                        with open(meta_file) as f:
                            info["measurement"] = json.load(f)
                    except Exception:
                        pass
                folders.append(info)
                if len(folders) >= 20:  # Only show recent 20
                    break
    return folders


RAW_CAPTURES_DIR = BASE_DIR / "raw_captures"
RAW_CAPTURES_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_raw_capture_dir(capture_dir: str) -> Path | None:
    """Find a capture directory inside raw_captures/ or directly in BASE_DIR."""
    # Strip any leading 'raw_captures/' if given
    clean_name = Path(capture_dir).name
    p1 = RAW_CAPTURES_DIR / clean_name
    if p1.exists() and p1.is_dir():
        return p1
    p2 = BASE_DIR / capture_dir
    if p2.exists() and p2.is_dir():
        return p2
    return None


def _get_next_raw_capture_name() -> str:
    """Find the highest numbered raw_capture_XX inside raw_captures/ and return raw_capture_{max+1:02d}."""
    max_idx = 0
    search_dirs = [RAW_CAPTURES_DIR, BASE_DIR]
    for sdir in search_dirs:
        if sdir.exists():
            for p in sdir.iterdir():
                if p.is_dir() and p.name.startswith("raw_capture_"):
                    m = re.search(r"raw_capture_(\d+)", p.name)
                    if m:
                        try:
                            max_idx = max(max_idx, int(m.group(1)))
                        except ValueError:
                            pass
    return f"raw_capture_{max_idx + 1:02d}"


def _get_raw_captures() -> list[str]:
    """Return sorted list of raw capture folder names."""
    captures = set()
    search_dirs = [RAW_CAPTURES_DIR, BASE_DIR]
    for sdir in search_dirs:
        if sdir.exists():
            for p in sdir.iterdir():
                if p.is_dir() and p.name.startswith("raw_capture_"):
                    captures.add(p.name)
    return sorted(list(captures))


def _get_capture_thumbnails(capture_dir: str, max_thumbs: int = 8) -> list[str]:
    """Return list of color frame filenames from a raw capture dir."""
    cap_path = _resolve_raw_capture_dir(capture_dir)
    if not cap_path:
        return []
    files = sorted(f.name for f in cap_path.iterdir() if f.name.endswith("_color.png"))
    return files[:max_thumbs]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    return jsonify({
        "running": _is_running,
        "elapsed": time.time() - _scan_start_time if _scan_start_time and _is_running else 0,
    })


@app.route("/api/next-raw-capture")
def api_next_raw_capture():
    return jsonify({"next_name": _get_next_raw_capture_name()})


@app.route("/api/raw-captures")
def api_raw_captures():
    return jsonify(_get_raw_captures())


@app.route("/api/scan-history")
def api_scan_history():
    return jsonify(_get_scan_folders())


@app.route("/api/capture-thumbs/<capture_name>")
def api_capture_thumbs(capture_name):
    return jsonify(_get_capture_thumbnails(capture_name))


@app.route("/raw-frame/<capture_name>/<filename>")
def raw_frame_image(capture_name, filename):
    cap_path = _resolve_raw_capture_dir(capture_name)
    if cap_path and (cap_path / filename).exists():
        return send_from_directory(str(cap_path), filename)
    return "Not found", 404


@app.route("/scan-file/<path:filepath>")
def scan_file(filepath):
    full = BASE_DIR / filepath
    if full.exists() and full.is_file():
        return send_file(str(full))
    return "Not found", 404


@app.route("/api/scan-results/<scan_name>")
def api_scan_results(scan_name):
    """Return detailed results from a scan folder."""
    scan_path = BASE_DIR / "stage_wise_output" / scan_name
    if not scan_path.exists():
        return jsonify({"error": "not found"}), 404

    result = {"name": scan_name, "files": []}

    # Check for measurement.json
    meas = scan_path / "measurement.json"
    if meas.exists():
        try:
            with open(meas) as f:
                result["measurement"] = json.load(f)
        except Exception:
            pass

    # List PLY files
    for f in sorted(scan_path.iterdir()):
        if f.is_file():
            result["files"].append({"name": f.name, "size": f.stat().st_size})

    return jsonify(result)


# ---------------------------------------------------------------------------
# Socket.IO events
# ---------------------------------------------------------------------------

def _build_command(config: dict) -> list[str]:
    """Build the command-line arguments directly with user pipeline flags."""
    cmd = [sys.executable, "main.py"]

    # Target (head or body)
    target = config.get("target", "head")
    cmd.extend(["--target", target])

    # User pipeline flags
    cmd.append("--face-quality")
    cmd.extend(["--gate-mode", "advisory"])
    cmd.append("--tsdf")
    cmd.append("--multiway")
    cmd.append("--loop-closure")

    # Save raw directory (automatically stored in raw_captures/)
    raw_dir = config.get("save_raw", "").strip()
    if not raw_dir:
        raw_dir = _get_next_raw_capture_name()

    # Prepend raw_captures/ if not already specified
    clean_parts = Path(raw_dir).parts
    if "raw_captures" not in clean_parts:
        save_path = f"raw_captures/{raw_dir}"
    else:
        save_path = raw_dir

    cmd.extend(["--save-raw", save_path])

    return cmd


def _stream_output(proc: subprocess.Popen, sid: str):
    """Read process stdout line-by-line and emit via SocketIO."""
    global _is_running, _scan_start_time
    stage_pattern = re.compile(r"Stage\s+(\d+)", re.IGNORECASE)
    current_stage = 0

    try:
        for raw_line in iter(proc.stdout.readline, ""):
            if not raw_line:
                break
            line = raw_line.rstrip("\n\r")

            # Detect stage transitions
            m = stage_pattern.search(line)
            if m:
                current_stage = int(m.group(1))
                socketio.emit("stage_update", {"stage": current_stage}, namespace="/")

            # Classify line type for coloured output
            line_type = "info"
            if "ERROR" in line or "Error" in line or "error" in line:
                line_type = "error"
            elif "WARNING" in line or "Warning" in line or "warn" in line:
                line_type = "warning"
            elif "Stage" in line or "=====" in line:
                line_type = "stage"
            elif "✓" in line or "PASS" in line or "saved" in line.lower() or "complete" in line.lower():
                line_type = "success"

            socketio.emit("log_line", {
                "text": line,
                "type": line_type,
                "stage": current_stage,
                "elapsed": time.time() - _scan_start_time if _scan_start_time else 0,
            }, namespace="/")

    except Exception as e:
        socketio.emit("log_line", {
            "text": f"[GUI] Stream error: {e}",
            "type": "error",
            "stage": current_stage,
            "elapsed": 0,
        }, namespace="/")

    proc.wait()
    _is_running = False
    exit_code = proc.returncode

    socketio.emit("scan_finished", {
        "exit_code": exit_code,
        "elapsed": time.time() - _scan_start_time if _scan_start_time else 0,
        "next_raw_capture": _get_next_raw_capture_name(),
    }, namespace="/")


@socketio.on("start_scan")
def handle_start_scan(config):
    global _process, _is_running, _scan_start_time

    with _process_lock:
        if _is_running:
            emit("log_line", {"text": "[GUI] A scan is already running!", "type": "error", "stage": 0, "elapsed": 0})
            return

        cmd = _build_command(config)
        cmd_str = " ".join(cmd)
        emit("log_line", {"text": f"[GUI] Starting: {cmd_str}", "type": "stage", "stage": 0, "elapsed": 0})
        emit("command_echo", {"command": cmd_str})

        try:
            _process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(BASE_DIR),
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            _is_running = True
            _scan_start_time = time.time()

            # Stream output in background thread
            t = threading.Thread(
                target=_stream_output,
                args=(_process, request.sid),
                daemon=True,
            )
            t.start()

            emit("scan_started", {"pid": _process.pid})

        except Exception as e:
            emit("log_line", {"text": f"[GUI] Failed to start: {e}", "type": "error", "stage": 0, "elapsed": 0})
            _is_running = False


@socketio.on("stop_scan")
def handle_stop_scan():
    global _process, _is_running
    with _process_lock:
        if _process and _is_running:
            _process.terminate()
            emit("log_line", {"text": "[GUI] Scan terminated by user.", "type": "warning", "stage": 0, "elapsed": 0})


@socketio.on("send_key")
def handle_send_key(data):
    """Send a keypress to the running subprocess stdin (not typically used since OpenCV captures keys)."""
    global _process
    if _process and _is_running and _process.stdin:
        try:
            key = data.get("key", "")
            _process.stdin.write(key + "\n")
            _process.stdin.flush()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    TEMPLATE_DIR.mkdir(exist_ok=True)
    STATIC_DIR.mkdir(exist_ok=True)
    print("\n" + "=" * 60)
    print("  RealSense 3D Scanner — Web GUI")
    print("  Open in browser: http://localhost:5050")
    print("=" * 60 + "\n")
    socketio.run(app, host="0.0.0.0", port=5050, debug=False, allow_unsafe_werkzeug=True)
