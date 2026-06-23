// Safe localStorage wrappers — private-browsing mode (iOS Safari / Firefox) throws
// SecurityError on access, so these swallow it and callers can use them unconditionally.
// A deliberate exception to routing browser-API access through PlatformAdapter.

export function lsGet(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

export function lsSet(key: string, val: string): void {
  try {
    localStorage.setItem(key, val);
  } catch {
    /* storage unavailable — ignore */
  }
}
