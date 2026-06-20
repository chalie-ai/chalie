// The Brain's own served origin — protocol + host + non-default port, NO
// trailing slash. This is what the app scans to know where to reach this
// instance. It is read from window.location.origin (never user-editable) and
// must parse as a valid absolute URL (spec §7 pairing gate). Throws on a
// non-parseable origin so the QR is never minted against garbage.
export function readBrainOrigin(): string {
  const origin = window.location.origin;
  // URL() throws on a malformed origin; "null" is the opaque-origin sentinel
  // (file://, sandboxed iframe) — both are unusable as a reachable host.
  if (!origin || origin === 'null') {
    throw new Error('This page has no reachable origin to link against.');
  }
  // Round-trips through URL to assert it parses; .origin is already
  // canonical (no trailing slash, default ports elided) so we return it as-is.
  new URL(origin);
  return origin;
}
