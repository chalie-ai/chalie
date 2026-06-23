/**
 * Brain API error-message helper.
 *
 * 401 handling now lives in the shared ApiClient (it redirects to /login/ on a
 * 401 unless the caller opts out), so there is no per-call withAuth() wrapper
 * here any more. This module only resolves user-facing error strings.
 */
import { HttpError } from '@chalie/shared';

/**
 * Resolve a user-facing message for a thrown API error.
 * - HTTP (non-2xx) error → the server's {error} message if present, else `httpFallback`.
 * - anything else (network/unexpected) → `networkFallback`.
 */
export function apiErrorMessage(e: unknown, httpFallback: string, networkFallback = 'Network error'): string {
  if (e instanceof HttpError) return e.error || httpFallback;
  return networkFallback;
}
