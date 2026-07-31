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

/**
 * Render a safe subset of Markdown to HTML for `v-html`. Escape-first, so any raw
 * HTML in the source is neutralised and only the tags produced here reach the DOM.
 * Links are protocol-gated to http(s)/mailto — anything else is left as literal text,
 * so `[x](javascript:…)` cannot inject. Covers headings, bold, italic, inline code,
 * links, bullet lists and paragraph breaks — the markdown LLM output actually uses.
 */
export function mdToHtml(md: string | null | undefined): string {
  if (!md) return '';
  return escapeHtml(md)
    .replace(/^#{1,6}\s+(.+)$/gm, '<h3>$1</h3>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*([^*\n]+)\*/g, '<em>$1</em>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    // (?=(X))\1 emulates an atomic group — JS has no possessive quantifiers, so this
    // prevents char-by-char backtracking while probing for the closing bracket.
    .replace(
      /\[(?=([^\]]+))\1\]\((?=((?:https?:\/\/|mailto:)[^)\s]+))\2\)/g,
      '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>',
    )
    .replace(/^[*-]\s+(.+)$/gm, '<li>$1</li>')
    .replace(/(<li>.*<\/li>\n?)+/g, (m) => `<ul>${m}</ul>`)
    .replace(/\n{2,}/g, '<br>');
}

/** Capitalize the first character of a string; empty input returns ''. */
export function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}
