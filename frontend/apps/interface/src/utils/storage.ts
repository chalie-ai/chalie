/**
 * Safe localStorage wrappers.
 * Private-browsing mode on iOS Safari / Firefox throws SecurityError on access;
 * these wrappers swallow that so callers can use them unconditionally.
 *
 * Note: most browser-API access goes through PlatformAdapter, but localStorage
 * for the theme pre-paint and host config are handled here and in shared/config/host.ts
 * as a deliberate exception (matches the P0 pattern).
 */

/** Return the stored value for key, or null when unavailable or on error. */
export function lsGet(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

/** Persist value under key. Silently ignores storage errors. */
export function lsSet(key: string, val: string): void {
  try {
    localStorage.setItem(key, val);
  } catch {
    /* storage unavailable — ignore */
  }
}
