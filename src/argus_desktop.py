"""A.R.G.U.S. desktop app: the dashboard in its own native window.

Run from the src folder:

    python argus_desktop.py

This starts the same backend as `python allnew.py` (one server on port 8000 that the
Android app also connects to at ws://<this PC>:8000/ws) and shows the dashboard in a
native window instead of a browser tab. On Windows the window uses Edge WebView2, which
ships with Windows 10 and 11. A loading screen shows each startup step while the models
load. Closing the window shuts the backend down.

If an A.R.G.U.S. server is already running on the port, the window simply attaches to it.

Options
    --browser   open the dashboard in the default browser instead of a window
    --debug     enable the WebView developer tools (right click > Inspect)
"""

import argparse
import json
import os
import socket
import sys
import threading
import time
import traceback
import webbrowser

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.getenv("ARGUS_PORT", "8000"))
URL = f"http://127.0.0.1:{PORT}/"
WS_MAX_SIZE = 32 * 1024 * 1024

SPLASH_HTML = """<!DOCTYPE html>
<html><head><meta charset="UTF-8"><style>
  html, body { height: 100%; margin: 0; }
  body { display: flex; align-items: center; justify-content: center;
         background: #f5f5f2; color: #1d1d1b;
         font-family: "Segoe UI", system-ui, -apple-system, sans-serif; }
  @media (prefers-color-scheme: dark) { body { background: #161615; color: #ececea; } }
  .box { width: min(520px, 86vw); text-align: center; }
  h1 { font-size: 28px; letter-spacing: 0.35em; font-weight: 600; margin: 0 0 6px; }
  .sub { opacity: 0.6; font-size: 13px; margin-bottom: 28px; }
  .bar { height: 3px; border-radius: 3px; background: rgba(127,127,127,0.2); overflow: hidden; }
  .bar i { display: block; height: 100%; width: 35%; background: currentColor;
           animation: slide 1.3s ease-in-out infinite; }
  @keyframes slide { from { transform: translateX(-100%); } to { transform: translateX(290%); } }
  #status { margin-top: 18px; font-size: 13px; opacity: 0.75; min-height: 1.4em;
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .err #status { white-space: pre-wrap; text-align: left; opacity: 1; color: #c0392b;
                 font-family: Consolas, monospace; font-size: 12px; }
  .err .bar { display: none; }
</style></head><body><div class="box" id="box">
  <h1>A.R.G.U.S.</h1>
  <div class="sub">Starting the multimodal engine</div>
  <div class="bar"><i></i></div>
  <div id="status">Loading models...</div>
</div><script>
  function setStatus(t) { document.getElementById('status').textContent = t; }
  function showError(t) { document.getElementById('box').className = 'box err'; setStatus(t); }
</script></body></html>"""


def port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


class SplashTee:
    """Copies console output to the real console and shows each line on the loading screen."""

    def __init__(self, stream, on_line):
        self.stream = stream
        self.on_line = on_line
        self.active = True
        self._buf = ""

    def write(self, text):
        self.stream.write(text)
        if self.active:
            self._buf += text
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                line = line.strip()
                if line:
                    try:
                        self.on_line(line)
                    except Exception:
                        pass
        return len(text)

    def flush(self):
        self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


class Backend:
    """Imports allnew (which loads the models) and runs its FastAPI app with uvicorn in a thread."""

    def __init__(self):
        self.server = None
        self.thread = None

    def start(self, timeout=60):
        os.chdir(BASE_DIR)  # allnew resolves model and log paths from the src folder
        if BASE_DIR not in sys.path:
            sys.path.insert(0, BASE_DIR)
        import uvicorn
        import allnew

        config = uvicorn.Config(allnew.app, host="0.0.0.0", port=PORT, ws_max_size=WS_MAX_SIZE)
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, name="argus-backend", daemon=True)
        self.thread.start()
        deadline = time.time() + timeout
        while not self.server.started:
            if not self.thread.is_alive():
                raise RuntimeError(f"The server stopped during startup. Is port {PORT} free?")
            if time.time() > deadline:
                raise RuntimeError("The server did not start in time.")
            time.sleep(0.1)

    def stop(self):
        if self.server is not None:
            self.server.should_exit = True
            if self.thread is not None:
                self.thread.join(timeout=5)


def run_browser():
    """Fallback: run the backend and open the dashboard in the default browser."""
    backend = Backend()
    if not port_in_use(PORT):
        backend.start()
    print(f"\n[SYSTEM] A.R.G.U.S. running at {URL}")
    webbrowser.open(URL)
    try:
        while backend.thread is None or backend.thread.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        backend.stop()


def run_window(debug=False):
    # WebView2 would otherwise ask for webcam/mic permission on every launch. The window only
    # ever shows our own local dashboard, so let it use the real devices without the prompt.
    extra = os.environ.get("WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS", "")
    if "--use-fake-ui-for-media-stream" not in extra:
        os.environ["WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS"] = (extra + " --use-fake-ui-for-media-stream").strip()

    import webview

    backend = Backend()
    attach = port_in_use(PORT)
    window = webview.create_window(
        "A.R.G.U.S.",
        url=URL if attach else None,
        html=None if attach else SPLASH_HTML,
        width=1440,
        height=900,
        min_size=(960, 640),
        background_color="#f5f5f2",
        text_select=True,
    )

    def boot():
        if attach:
            print(f"[SYSTEM] Attached to the A.R.G.U.S. server already running on port {PORT}.")
            return
        tee = SplashTee(sys.stdout, lambda line: window.evaluate_js(f"setStatus({json.dumps(line)})"))
        sys.stdout = tee
        try:
            backend.start()
            tee.active = False
            print(f"\n[SYSTEM] A.R.G.U.S. running at {URL} (desktop window)")
            window.load_url(URL)
        except Exception as exc:
            tee.active = False
            traceback.print_exc()
            message = f"A.R.G.U.S. could not start:\n{type(exc).__name__}: {exc}\n\nSee the console for details."
            window.evaluate_js(f"showError({json.dumps(message)})")

    webview.start(boot, debug=debug, private_mode=False)

    # The window was closed: stop the server, then exit even if mic or model threads are still busy.
    backend.stop()
    os._exit(0)


def main():
    parser = argparse.ArgumentParser(description="A.R.G.U.S. desktop app")
    parser.add_argument("--browser", action="store_true", help="open in the default browser instead of a window")
    parser.add_argument("--debug", action="store_true", help="enable WebView developer tools")
    args = parser.parse_args()

    if args.browser:
        run_browser()
        return
    try:
        import webview  # noqa: F401
    except ImportError:
        print("[WARNING] pywebview is not installed (pip install pywebview). Opening the browser instead.")
        run_browser()
        return
    run_window(debug=args.debug)


if __name__ == "__main__":
    main()
