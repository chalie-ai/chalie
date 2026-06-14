/**
 * Format a backend timestamp string to `YYYY-MM-DD HH:MM` in local time.
 *
 * Verbatim port of legacy `frontend/brain/app.js:385-393` (`BrainApp.formatDate`),
 * used by the Cognition / Scheduler / Documents / Skills / Policies panels:
 *   - empty / falsy input  → ''
 *   - a naive timestamp (no `T`, `+`, or `Z`) is treated as UTC: ' ' → 'T', then append 'Z'
 *   - an unparseable value  → returned as-is
 */
export function formatDate(raw: string | null | undefined): string {
  if (!raw) return '';
  try {
    const d = new Date(
      raw.includes('T') || raw.includes('+') || raw.includes('Z') ? raw : raw.replace(' ', 'T') + 'Z',
    );
    if (isNaN(d.getTime())) return raw;
    const pad = (n: number): string => String(n).padStart(2, '0');
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
  } catch {
    return raw;
  }
}

/**
 * Escape HTML special characters. Verbatim port of legacy
 * `frontend/brain/app.js:380-383` (`BrainApp.escapeHtml`). Needed only where
 * markup is assembled for `v-html` (e.g. the World view's markdown renderer);
 * ordinary `{{ }}` interpolation already escapes its content.
 */
export function escapeHtml(s: string | null | undefined): string {
  if (!s) return '';
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
