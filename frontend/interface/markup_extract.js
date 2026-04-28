// Extract plaintext from XML markup for the speak button (TTS).
// Drops <actions> entirely. Uses <img alt> when present.

// Bound input to avoid pathological regex backtracking on hostile input.
// Real content is bounded by LLM token limits; 5 MB is generous.
const MAX_CONTENT_LEN = 5_000_000;

const TAG_RE = /<\s*(\/)?\s*([a-zA-Z][a-zA-Z0-9]*)\s*([^>]*?)\s*(\/)?\s*>/g;
const ALT_RE = /\balt\s*=\s*"([^"]*)"/i;

const DROP_TAGS = new Set(['actions']);

// Single shared DOMParser. Parsing as text/html lets us delegate ALL XML and
// numeric character entity decoding to the browser — including `&#9731;`,
// `&#x2603;`, `&hearts;`, etc. Manual maps would silently miss numerics and
// every named entity outside the obvious five.
const _parser = (typeof DOMParser !== 'undefined') ? new DOMParser() : null;

function decodeEntities(text) {
  if (text.indexOf('&') === -1) return text;
  if (_parser) {
    try {
      // Wrap in <body> so leading whitespace is preserved verbatim.
      const doc = _parser.parseFromString(`<!doctype html><body>${text}`, 'text/html');
      return doc.body.textContent || '';
    } catch (_e) { /* fall through */ }
  }
  // Fallback (no DOM available — Node test env). Covers the common five.
  return text
    .replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"').replace(/&#39;/g, "'");
}

export function extractPlaintext(content) {
  if (!content) return '';
  if (content.length > MAX_CONTENT_LEN) {
    content = content.slice(0, MAX_CONTENT_LEN);
  }
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
  return decodeEntities(out.join(' '))
    .replace(/\s+/g, ' ')
    .trim();
}
