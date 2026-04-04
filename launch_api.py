"""
Start the Medical AI API and open Swagger in your browser.

Run from the project folder (double-click may not set the folder — use terminal):

    cd path\\to\\VIT_SET_Project
    python launch_api.py

Keep this window open. Closing it stops the server (connection refused in browser).
"""
from __future__ import annotations

import os
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MEDICAL_AI_SKIP_RAG", "1")
os.environ["PYTHONPATH"] = str(ROOT)

HOST = "0.0.0.0"  # listen on all interfaces; use http://127.0.0.1:8000/docs in browser
PORT = 8000
URL = f"http://127.0.0.1:{PORT}/docs"


def _open_browser() -> None:
    time.sleep(2.5)
    print(f"\n>>> Opening browser: {URL}\n")
    webbrowser.open(URL)


def main() -> None:
    try:
        import uvicorn
    except ImportError:
        print("ERROR: uvicorn not installed.\nRun:  python -m pip install -r requirements.txt")
        sys.exit(1)

    print("=" * 60)
    print("Medical AI API")
    print(f"  Server:  http://127.0.0.1:{PORT}   (also try http://localhost:{PORT})")
    print(f"  Swagger: {URL}")
    print("  Leave this window OPEN while you use the site.")
    print("=" * 60)

    threading.Thread(target=_open_browser, daemon=True).start()

    try:
        uvicorn.run(
            "src.api.main:app",
            host=HOST,
            port=PORT,
            reload=False,
        )
    except OSError as e:
        if "10048" in str(e) or "address already in use" in str(e).lower():
            print(f"\nERROR: Port {PORT} is already in use. Close the other app or change PORT in launch_api.py.\n")
        raise


if __name__ == "__main__":
    main()
