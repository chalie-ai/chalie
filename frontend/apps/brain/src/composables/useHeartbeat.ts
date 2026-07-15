/**
 * Session heartbeat: polls GET /auth/status every 5 min and on tab refocus.
 * Two redirect conditions, distinct since has_session was unmasked from
 * vault_state (#1878 part 2):
 *   - has_session && vault_state==='locked' → vault re-sealed after a backend
 *     restart. Brain has no unlock UI, so hand off to the interface app whose
 *     UnlockVault overlay unseals in place.
 *   - !has_session → genuine session expiry → /login/.
 * Module-scoped state makes useHeartbeat() a process-wide singleton.
 */

import { system } from '../api';

let _intervalId: ReturnType<typeof setInterval> | null = null;
let _redirected = false;

function _redirect(to: string): void {
  if (_redirected) return;
  _redirected = true;
  stop();
  window.location.replace(to);
}

async function _checkSession(): Promise<void> {
  if (_redirected) return;
  try {
    const data = await system.authStatus();
    if (!data.has_master_account) return;
    if (data.has_session && data.vault_state === 'locked') {
      _redirect('/');
      return;
    }
    if (!data.has_session) {
      _redirect(
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
