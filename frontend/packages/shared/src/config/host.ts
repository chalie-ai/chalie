const HOST_KEY = 'chalie_backend_host';

/** Returns the configured backend host, or '' for same-origin. */
export function getHost(): string {
  try {
    return localStorage.getItem(HOST_KEY) ?? '';
  } catch {
    return '';
  }
}

/** Persist (or clear, when empty) the backend host override. */
export function setHost(host: string): void {
  try {
    if (host) localStorage.setItem(HOST_KEY, host);
    else localStorage.removeItem(HOST_KEY);
  } catch {
    /* storage unavailable — fall back to same-origin */
  }
}
