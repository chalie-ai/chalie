import { createRouter, createWebHistory } from 'vue-router';
import { AuthError, getToken, HttpError, isTauri } from '@chalie/shared';
import HomeView from './views/HomeView.vue';
import { system } from './api/system';

export const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [{ path: '/', name: 'home', component: HomeView }],
});

// Whether the auth gate issued a hard redirect on the initial navigation.
// main.ts reads this (after router.isReady()) to decide whether to mount — gating
// the mount keeps the WebSocket from connecting when the gate is navigating away.
let _gateRedirected = false;
export function authGateRedirected(): boolean {
  return _gateRedirected;
}

function redirect(to: string): false {
  _gateRedirected = true;
  window.location.replace(to);
  return false;
}

// Auth gate — runs once on the initial navigation to `/`. On a missing
// account/session/providers it hard-redirects to the appropriate route.
// Error handling:
//   - Reachable server with a non-ok status (HttpError/AuthError) → treated as
//     all-false, which resolves to the onboarding redirect below.
//   - Genuine network failure → stay; the backend guards the real endpoints.
router.beforeEach(async () => {
  // Native runtime with no bearer yet → pair by QR (no cookie-login on mobile).
  // Pairing supplies the backend host, so nothing can be probed until then — this
  // MUST precede authStatus (a host-less probe rejects with a network TypeError,
  // which the catch below treats as "stay", stranding the unpaired app on the chat
  // shell). /pairing/ is its own page (like /login/), so this full-document
  // redirect lands there without re-entering this gate. Web, or a token already
  // present, skips this.
  if (isTauri && !getToken()) {
    return redirect('/pairing/');
  }

  let status;
  try {
    status = await system.authStatus();
  } catch (err) {
    if (err instanceof HttpError || err instanceof AuthError) {
      return redirect('/on-boarding/');
    }
    return true;
  }

  const { has_master_account, has_session, has_providers } = status;

  if (!has_master_account) {
    return redirect('/on-boarding/');
  }
  if (!has_session) {
    return redirect(
      '/login/?next=' + encodeURIComponent(window.location.pathname + window.location.search),
    );
  }
  if (!has_providers) {
    return redirect('/brain/');
  }
  return true;
});
