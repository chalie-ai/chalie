import { createRouter, createWebHistory } from 'vue-router';
import { HttpError, AuthError } from '@chalie/shared';
import HomeView from './views/HomeView.vue';
import { system } from './api/system';

export const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [{ path: '/', name: 'home', component: HomeView }],
});

/**
 * Whether the auth gate issued a hard redirect on the initial navigation.
 *
 * main.ts reads this (after `router.isReady()`) to decide whether to mount the
 * app at all. Gating the mount keeps the WebSocket from connecting when the gate
 * is about to navigate the page away — parity with legacy app.js:58
 * (`await chalieGateReady; if (!gate.stay) return;`), which wired *nothing*
 * until the gate resolved to "stay".
 */
let _gateRedirected = false;
export function authGateRedirected(): boolean {
  return _gateRedirected;
}

function redirect(to: string): false {
  _gateRedirected = true;
  window.location.replace(to);
  return false;
}

/**
 * Auth gate — port of frontend/shared/auth-gate.js `chat` page branch.
 *
 * Runs once on the initial navigation to `/`. On missing account/session/
 * providers it performs a hard redirect to the appropriate legacy route.
 *
 * Error handling mirrors legacy auth-gate.js:22-27 exactly:
 *   - A reachable server that answers with a non-ok status (HttpError, incl.
 *     500; AuthError, i.e. 401) is treated as "all false" → the chain below
 *     sends the user to /on-boarding/.
 *   - A genuine network failure (fetch rejects, no response) → stay; the
 *     backend guards the real endpoints anyway.
 */
router.beforeEach(async () => {
  let status;
  try {
    status = await system.authStatus();
  } catch (err) {
    if (err instanceof HttpError || err instanceof AuthError) {
      // Server responded, but not ok — legacy substitutes all-false, which
      // resolves to the onboarding redirect (has_master_account === false).
      return redirect('/on-boarding/');
    }
    // Network / server unreachable — stay; server guards anyway.
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
