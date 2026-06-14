/**
 * Session heartbeat composable.
 *
 * Port of legacy app.js _checkSession / _startHeartbeat (lines 311-333):
 *   - Polls GET /auth/status every 5 minutes.
 *   - Also fires on visibilitychange → visible (tab comes back to foreground).
 *   - On a non-authed response (has_master_account && !has_session) → hard
 *     redirect to /login/?next=<path>.
 *
 * One singleton instance is shared across calls (useHeartbeat() always returns
 * the same object, matching the legacy single-instance pattern).
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
    // Transient network error — ignore, try again next tick.
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
