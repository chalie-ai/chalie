"""Instant holding page while the real backend boots.

``run.py`` binds this stdlib-only HTTP server on the public port *before* the
heavy imports (numpy/transformers), schema convergence, and startup migrations
that delay Flask's bind — seconds on fast hardware, minutes on slow machines
or first-run installs. Docker publishes the container port immediately, so
without a listener every early connection is refused or reset: browsers show
a blank page and a console full of failed fetches.

While active, every GET/POST/HEAD is answered ``503`` (other verbs get the
stdlib's 501 — nothing legitimate sends them this early):

- Requests with ``Accept: text/html`` (browser navigations) get a
  self-refreshing holding page that polls ``/ready`` and reloads into the
  real app the moment it answers.
- Everything else (SPA fetches, health probes) gets
  ``{"ready": false, "status": "starting"}`` — a clean "not yet" instead of a
  dropped connection.

HTTP only: TLS state lives in the database, which is not readable this early
in boot. On an SSL-enabled install the pre-bind window behaves as before this
existed (handshake failure) — no worse, and the common case is fixed.
"""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_STATUS_BODY = json.dumps({"ready": False, "status": "starting"}).encode()

# Colors mirror the dark/light theme tokens in
# frontend/packages/shared/src/styles/_tokens.scss (--bg / --text / --violet);
# this page is served before any frontend bundle exists, so they are inlined.
_PAGE = b"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chalie &mdash; starting</title>
<style>
  :root{--bg:#07070b;--text:#eae6f2;--muted:rgba(234,230,242,.38);--accent:#8A5CFF}
  @media (prefers-color-scheme:light){
    :root{--bg:#F6F4F1;--text:#1A1626;--muted:rgba(26,22,38,.45);--accent:#6E3DEB}
  }
  html,body{height:100%;margin:0}
  body{display:flex;align-items:center;justify-content:center;
    background:var(--bg);color:var(--text);
    font:16px/1.5 system-ui,-apple-system,sans-serif}
  main{text-align:center;padding:2rem}
  .spinner{width:36px;height:36px;margin:0 auto 1.5rem;
    border:3px solid var(--muted);border-top-color:var(--accent);
    border-radius:50%;animation:spin 1s linear infinite}
  @keyframes spin{to{transform:rotate(360deg)}}
  h1{font-size:1.35rem;margin:0 0 .5rem;font-weight:600}
  p{margin:0 auto;color:var(--muted);max-width:34ch}
</style>
</head>
<body>
<main>
  <div class="spinner" role="status" aria-label="Loading"></div>
  <h1>Chalie is starting</h1>
  <p>Setting things up &mdash; the first start can take a few minutes.
     This page refreshes automatically.</p>
</main>
<script>
(async function poll(){
  try{
    const r = await fetch("/ready",{cache:"no-store"});
    if(r.ok){location.reload();return}
  }catch{}
  setTimeout(poll,2000);
})();
</script>
</body>
</html>
"""


class _Handler(BaseHTTPRequestHandler):
    def _answer(self, with_body: bool = True) -> None:
        wants_html = "text/html" in (self.headers.get("Accept") or "")
        body = _PAGE if wants_html else _STATUS_BODY
        self.send_response(503)
        self.send_header(
            "Content-Type",
            "text/html; charset=utf-8" if wants_html else "application/json",
        )
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Retry-After", "2")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if with_body:
            self.wfile.write(body)

    def do_GET(self) -> None:
        self._answer()

    def do_POST(self) -> None:
        self._answer()

    def do_HEAD(self) -> None:
        self._answer(with_body=False)

    def log_message(self, format: str, *args: object) -> None:
        pass  # holding-page polls are noise


class BootScreen:
    """Owns ``host:port`` from process start until the real server takes over."""

    def __init__(self, host: str, port: int) -> None:
        self._host = host
        self._port = port
        self._server: ThreadingHTTPServer | None = None

    def start(self) -> None:
        try:
            self._server = ThreadingHTTPServer((self._host, self._port), _Handler)
        except OSError as exc:
            # Port already taken — the real server's bind will surface it loudly.
            print(f"[BOOT] holding page skipped: {exc}", file=sys.stderr, flush=True)
            return
        threading.Thread(
            target=self._server.serve_forever, name="boot-screen", daemon=True
        ).start()

    def stop(self) -> None:
        """Release the port; the API worker calls this right before it binds.

        Re-entered on every worker crash-restart (WorkerManager re-invokes the
        Flask worker from scratch), so the ``None`` guard is load-bearing.
        """
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        self._server = None
