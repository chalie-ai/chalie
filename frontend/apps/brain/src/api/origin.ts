// The Brain's own served origin — protocol + host + non-default port, NO
// trailing slash. This is what the device scans to know where to reach this
// instance; it is read from the browser location (never user-editable) and is
// already canonical (default ports elided), so it is returned verbatim.
export function readBrainOrigin(): string {
  const origin = globalThis.location.origin;
  // "null" is the opaque-origin sentinel (file://, sandboxed iframe); an empty
  // or opaque origin is unusable as a reachable host, so refuse to mint a QR.
  if (!origin || origin === 'null') {
    throw new Error('This page has no reachable origin to link against.');
  }
  return origin;
}
