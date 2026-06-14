/**
 * Brain API HTTP helper.
 *
 * Centralises 401 handling: the shared ApiClient throws AuthError on 401
 * without redirecting. This module catches it and redirects to /login/.
 * Port of legacy BrainApp.apiFetch()'s `if (res.status === 401)` branch
 * (app.js:68) — but targeted at /brain-next/ (not /brain/).
 */
import { AuthError } from '@chalie/shared';

export function handle401(err: unknown): never {
  if (err instanceof AuthError) {
    window.location.replace(
      '/login/?next=' + encodeURIComponent(window.location.pathname + window.location.search),
    );
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
