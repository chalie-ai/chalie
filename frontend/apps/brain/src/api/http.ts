/**
 * Brain API HTTP helper.
 *
 * Centralises 401 handling: the shared ApiClient throws AuthError on 401
 * without redirecting. This module catches it and redirects to /login/.
 * Port of legacy BrainApp.apiFetch()'s `if (res.status === 401)` branch
 * (app.js:68) — but targeted at /brain-next/ (not /brain/).
 */
import { AuthError } from '@chalie/shared';

/** Redirect to /login/ preserving the current path and query string as `next`. */
export function redirectToLogin(): never {
  const next = window.location.pathname + window.location.search;
  window.location.replace('/login/?next=' + encodeURIComponent(next));
  throw new Error('redirecting to login');
}

export function handle401(err: unknown): never {
  if (err instanceof AuthError) {
    redirectToLogin();
  }
  throw err;
}

/** Wrap an api call and redirect on 401. */
export async function withAuth<T>(fn: () => Promise<T>): Promise<T> {
  try {
    return await fn();
  } catch (err) {
    return handle401(err);
  }
}
