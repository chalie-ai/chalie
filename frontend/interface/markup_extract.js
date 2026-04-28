// Extract plaintext from XML markup for the speak button (TTS).
// Drops <actions> entirely. Uses <img alt> when present.

const TAG_RE = /<\s*(\/)?\s*([a-zA-Z][a-zA-Z0-9]*)\s*([^>]*?)\s*(\/)?\s*>/g;
const ALT_RE = /\balt\s*=\s*"([^"]*)"/i;

const DROP_TAGS = new Set(['actions']);

export function extractPlaintext(content) {
  if (!content) return '';
  const out = [];
  let skipDepth = 0;
  let pos = 0;
  TAG_RE.lastIndex = 0;
  let m;
  while ((m = TAG_RE.exec(content)) !== null) {
    if (skipDepth === 0 && m.index > pos) {
      out.push(content.slice(pos, m.index));
    }
    const isClose = !!m[1];
    const name = m[2].toLowerCase();
    const isVoid = !!m[4];
    if (DROP_TAGS.has(name)) {
      if (isClose) skipDepth = Math.max(0, skipDepth - 1);
      else if (!isVoid) skipDepth += 1;
    } else if (skipDepth === 0 && name === 'img' && isVoid) {
      const altMatch = m[3].match(ALT_RE);
      if (altMatch && altMatch[1]) out.push(altMatch[1]);
    }
    pos = m.index + m[0].length;
  }
  if (skipDepth === 0 && pos < content.length) {
    out.push(content.slice(pos));
  }
  return out
    .join(' ')
    .replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&#39;/g, "'")
    .replace(/\s+/g, ' ')
    .trim();
}
