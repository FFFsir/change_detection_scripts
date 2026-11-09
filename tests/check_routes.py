#!/usr/bin/env python
"""Route smoke check: start the app and require 200 from every page.

The unittest suite covers the data and maths layers; this script covers the
thing those tests cannot: that NiceGUI can actually build all three pages
against the configured packages.

    .venv/Scripts/python.exe tests/check_routes.py [--timeout 60]

The server is started on a free local port and stopped again afterwards. It is
a read-only check: no data file is written.
"""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
APP = REPO_ROOT / "app" / "change_detection_webui.py"
ROUTES = ("/", "/aef", "/dw")


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for(port: int, deadline: float) -> bool:
    url = f"http://127.0.0.1:{port}/"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, OSError, TimeoutError):
            time.sleep(0.5)
    return False


def check(port: int) -> int:
    failures = 0
    for route in ROUTES:
        url = f"http://127.0.0.1:{port}{route}"
        try:
            with urllib.request.urlopen(url, timeout=20) as response:
                status = response.status
                body = response.read()
        except (urllib.error.URLError, OSError, TimeoutError) as error:
            print(f"[FAIL] GET {route}  {error}")
            failures += 1
            continue
        ok = status == 200 and b"AEF" in body
        print(f"[{'PASS' if ok else 'FAIL'}] GET {route}  status={status} bytes={len(body)}")
        if not ok:
            failures += 1
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args()

    port = free_port()
    process = subprocess.Popen(
        [sys.executable, str(APP), "--port", str(port)],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        if not wait_for(port, time.time() + args.timeout):
            print("[FAIL] server did not become ready")
            output = process.stdout.read() if process.stdout else ""
            print(output[-2000:])
            return 1
        failures = check(port)
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()

    if failures:
        print(f"未通过项：{failures}")
        return 1
    print("所有页面均可访问。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
