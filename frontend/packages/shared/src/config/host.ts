const HOST_KEY = 'chalie_backend_host';
const TOKEN_KEY = 'chalie_access_token';
const USERNAME_KEY = 'chalie_username';

/** Configured backend host, or '' for same-origin. */
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

/** Raw bearer token from pairing, or '' when unpaired (web path). */
export function getToken(): string {
  try {
    return localStorage.getItem(TOKEN_KEY) ?? '';
  } catch {
    return '';
  }
}

/** Persist (or clear, when empty) the bearer access token. */
export function setToken(token: string): void {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token);
    else localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* storage unavailable — token attach becomes a no-op */
  }
}

/** Master LOGIN username from the scanned QR, or '' when unpaired. */
export function getUsername(): string {
  try {
    return localStorage.getItem(USERNAME_KEY) ?? '';
  } catch {
    return '';
  }
}

/** Persist (or clear, when empty) the master login username from pairing. */
export function setUsername(username: string): void {
  try {
    if (username) localStorage.setItem(USERNAME_KEY, username);
    else localStorage.removeItem(USERNAME_KEY);
  } catch {
    /* storage unavailable — username is lost; the user must re-pair */
  }
}
