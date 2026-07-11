/** True only for absolute http(s) URLs. Base-less ``new URL()`` throws on
 * relative, scheme-less, and protocol-relative input, so those are refused —
 * unlike an anchor-probe (``a.href = ...``) which resolves them against the
 * page origin and would wrongly pass them through.
 */
export function isAbsoluteHttpUrl(url: string): boolean {
  try {
    const { protocol } = new URL(url);
    return protocol === 'http:' || protocol === 'https:';
  } catch {
    return false;
  }
}
