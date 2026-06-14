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
