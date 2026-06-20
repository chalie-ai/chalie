/**
 * Backend timestamp → `YYYY-MM-DD HH:MM` local time. Falsy → ''; naive stamps
 * (no `T`/`+`/`Z`) are treated as UTC; an unparseable value is returned as-is.
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
 * Escape HTML special chars. Needed only where markup is assembled for `v-html`;
 * ordinary `{{ }}` interpolation already escapes its content.
 */
export function escapeHtml(s: string | null | undefined): string {
  if (!s) return '';
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
