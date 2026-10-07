"""Instant holding page while the real backend boots.

``run.py`` binds this stdlib-only HTTP server on the public port *before* the
heavy imports (numpy, ONNX Runtime), database provisioning, and startup migrations
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

A boot that cannot proceed calls :meth:`BootScreen.fail`, which flips the same
listener from "starting" to a terminal error page naming what is missing. The
holding page's job is to say *wait*; the failure page's job is to say *stop, and
here is why* — a boot blocker the operator can only see in container logs is the
same silent failure this file exists to prevent. The failure page does not poll:
nothing is coming, so a spinner would lie.
"""

import base64
import html
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_STATUS_BODY = json.dumps({"ready": False, "status": "starting"}).encode()

# Set by BootScreen.fail() — read by every handler thread. Assignment of a str
# is atomic under the GIL, and the transition is one-way (starting -> failed),
# so no lock is needed.
_FAILURE: str | None = None

# Colors mirror the Afterhours tokens in
# frontend/packages/shared/src/styles/afterhours.css, and the pulsing dot matches
# the interface's own loading screen; this page is served before any frontend
# bundle exists, so they are inlined. The Afterhours fonts are embedded from the
# built interface bundle for the same reason — this listener answers every
# request with the page itself and cannot serve font files.


def _font_face_css() -> str:
    """Return @font-face rules with base64 ``data:font/woff2`` URLs.

    Read once from the built interface bundle at import. Vite hashes the asset
    names, so each file is globbed and the first match taken; a missing file
    contributes no rule, and the page falls back to the system fonts already in
    each stack.
    """
    assets = (
        Path(__file__).resolve().parents[1]
        / "frontend" / "apps" / "interface" / "dist" / "assets"
    )
    # Family names and weight ranges mirror afterhours.css.
    rules = []
    for family, pattern, weight in (
        (
            "Epilogue Variable",
            "epilogue-latin-wght-normal-*.woff2",
            "100 900",
        ),
        (
            "Instrument Sans Variable",
            "instrument-sans-latin-wght-normal-*.woff2",
            "400 700",
        ),
        (
            "JetBrains Mono Variable",
            "jetbrains-mono-latin-wght-normal-*.woff2",
            "100 800",
        ),
    ):
        matches = sorted(assets.glob(pattern))
        if not matches:
            continue
        data = base64.b64encode(matches[0].read_bytes()).decode("ascii")
        rules.append(
            "@font-face{font-family:'%s';font-style:normal;font-display:swap;"
            "font-weight:%s;src:url('data:font/woff2;base64,%s')"
            " format('woff2-variations')}" % (family, weight, data)
        )
    return "\n".join(rules)


_FONTS = _font_face_css()
_FONTS_BYTES = _FONTS.encode("ascii")

_PAGE = (
    b"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chalie &mdash; starting</title>
<style>
  """
    + _FONTS_BYTES
    + b"""
  :root{--bg:#0E0B10;--text:#F5F0F7;--muted:#A79FAF;--pink:#FF4FA3}
  @media (prefers-color-scheme:light){
    :root{--bg:#F5F0F7;--text:#0E0B10;--muted:#5B5363}
  }
  html,body{height:100%;margin:0}
  body{display:flex;align-items:center;justify-content:center;
    background:var(--bg);color:var(--text);
    font:15px/1.55 'Instrument Sans Variable',system-ui,-apple-system,sans-serif}
  main{text-align:center;padding:2rem}
  .dot{width:16px;height:16px;margin:0 auto 1.5rem;
    background:var(--pink);animation:working 1.4s ease-in-out infinite}
  @keyframes working{50%{opacity:.3}}
  @media (prefers-reduced-motion:reduce){.dot{animation:none}}
  h1{font-family:'Epilogue Variable',system-ui,sans-serif;
    font-size:1.25rem;margin:0 0 .5rem;font-weight:800;letter-spacing:-.02em}
  p{margin:0 auto;color:var(--muted);max-width:34ch}
</style>
</head>
<body>
<main>
  <div class="dot" role="status" aria-label="Loading"></div>
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
)


# Same tokens as _PAGE, no pulse and no poll — this state is terminal. The detail
# sits in a status panel: surface ground, text in the deny colour.
# %s is the escaped detail (what is missing).
_FAIL_PAGE = (
    """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chalie &mdash; failed to start</title>
<style>
  """
    + _FONTS
    + """
  :root{--bg:#0E0B10;--surface:#1B1620;--text:#F5F0F7;--muted:#A79FAF;--deny-text:#FF6B5E}
  @media (prefers-color-scheme:light){
    :root{--bg:#F5F0F7;--surface:#FFFFFF;--text:#0E0B10;--muted:#5B5363;--deny-text:#B83226}
  }
  html,body{height:100%%;margin:0}
  body{display:flex;align-items:center;justify-content:center;
    background:var(--bg);color:var(--text);
    font:15px/1.55 'Instrument Sans Variable',system-ui,-apple-system,sans-serif}
  main{text-align:center;padding:2rem;max-width:52ch}
  h1{font-family:'Epilogue Variable',system-ui,sans-serif;
    font-size:1.25rem;margin:0 0 .75rem;font-weight:800;letter-spacing:-.02em}
  code{display:block;margin:0 0 .75rem;padding:8px 16px;
    background:var(--surface);color:var(--deny-text);
    font:12px/1.5 'JetBrains Mono Variable',ui-monospace,SFMono-Regular,Menlo,monospace;
    word-break:break-word;text-align:left}
  p{margin:0;color:var(--muted)}
</style>
</head>
<body>
<main>
  <h1>Chalie failed to load</h1>
  <code>%s</code>
  <p>Chalie stopped before starting rather than running degraded.
     Install the missing dependency and start Chalie again.</p>
</main>
</body>
</html>
"""
)


class _Handler(BaseHTTPRequestHandler):
    def _answer(self, with_body: bool = True) -> None:
        wants_html = "text/html" in (self.headers.get("Accept") or "")
        if _FAILURE is not None:
            body = (
                _FAIL_PAGE % html.escape(_FAILURE)
            ).encode() if wants_html else json.dumps(
                {"ready": False, "status": "failed", "error": _FAILURE},
            ).encode()
        else:
            body = _PAGE if wants_html else _STATUS_BODY
        self.send_response(503)
        self.send_header(
            "Content-Type",
            "text/html; charset=utf-8" if wants_html else "application/json",
        )
        self.send_header("Content-Length", str(len(body)))
        if _FAILURE is None:
            # Only promise a retry while one is actually coming.
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

    def fail(self, detail: str) -> None:
        """Flip the holding page to a terminal failure page naming ``detail``.

        One-way: there is no path back to "starting". The caller is expected to
        stop booting straight after — the listener stays up purely so the
        browser gets an answer instead of a refused connection, which is the
        difference between an operator seeing the cause and seeing nothing.
        """
        global _FAILURE  # noqa: PLW0603 — module-level page state, one-way flip
        _FAILURE = detail

    def serve_forever(self) -> None:
        """Block on the holding/failure page until the process is killed.

        Only used on the failure path, where there is no real server to hand the
        port to: without this ``run.py`` would fall off the end of ``main`` and
        the daemon thread would die with it, refusing the very connection the
        failure page exists to answer.
        """
        if self._server is None:
            return
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass

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
