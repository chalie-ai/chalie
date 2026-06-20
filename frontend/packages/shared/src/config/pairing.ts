/**
 * QR pairing payload (spec §7). `host` is the Brain's own `window.location.origin`
 * (protocol + host + non-default port, no trailing slash); `token` is the raw
 * bearer minted once by POST /api/wrappers; `username` is the master login
 * username read from GET /auth/username so UnlockVault needs only a password.
 */
export interface PairingPayload {
  v: 1;
  host: string;
  token: string;
  username: string;
}

/**
 * Validate an untrusted decoded QR value against the v1 contract: `v === 1`,
 * `host` parses as an absolute URL, `token` and `username` non-empty. Throws on
 * any violation; returns the narrowed payload on success. Used by both the
 * generator (fail loud before rendering a QR) and the scanner (reject a garbage scan).
 */
export function validatePairingPayload(value: unknown): PairingPayload {
  if (typeof value !== 'object' || value === null) throw new Error('Pairing payload is not an object.');
  const p = value as Record<string, unknown>;
  if (p.v !== 1) throw new Error('Unsupported pairing version.');
  if (typeof p.host !== 'string' || p.host.length === 0) throw new Error('Missing host.');
  try { new URL(p.host); } catch { throw new Error('Host is not a valid URL.'); }
  if (typeof p.token !== 'string' || p.token.length === 0) throw new Error('Missing token.');
  if (typeof p.username !== 'string' || p.username.length === 0) throw new Error('Missing username.');
  return { v: 1, host: p.host, token: p.token, username: p.username };
}
