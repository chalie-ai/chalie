/**
 * Session heartbeat: polls GET /auth/status every 5 min and on tab refocus;
 * a logged-out response (has_master_account && !has_session) hard-redirects to
 * /login. Module-scoped state makes useHeartbeat() a process-wide singleton.
 */

let _intervalId: ReturnType<typeof setInterval> | null = null;
let _redirected = false;

async function _checkSession(): Promise<void> {
  if (_redirected) return;
  try {
    const res = await fetch('/auth/status', { credentials: 'same-origin' });
    if (!res.ok) return;
    const data = (await res.json()) as {
      has_master_account: boolean;
      has_session: boolean;
    };
    if (data.has_master_account && !data.has_session) {
      _redirected = true;
      stop();
      window.location.replace(
        '/login/?next=' +
          encodeURIComponent(window.location.pathname + window.location.search),
      );
    }
  } catch {
    // Transient network error — retry next tick.
  }
}

function _onVisibility(): void {
  if (document.visibilityState === 'visible') {
    void _checkSession();
  }
}

function start(): void {
  if (_intervalId !== null) return;
  _intervalId = setInterval(() => void _checkSession(), 5 * 60 * 1000);
  document.addEventListener('visibilitychange', _onVisibility);
}

function stop(): void {
  if (_intervalId !== null) {
    clearInterval(_intervalId);
    _intervalId = null;
  }
  document.removeEventListener('visibilitychange', _onVisibility);
}

export function useHeartbeat() {
  return { start, stop };
}
