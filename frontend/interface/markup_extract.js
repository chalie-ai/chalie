// Extract plaintext from XML markup for the speak button (TTS).
// Drops <actions> entirely. Uses <img alt> when present.

// Bound input to avoid pathological regex backtracking on hostile input.
// Real content is bounded by LLM token limits; 5 MB is generous.
const MAX_CONTENT_LEN = 5_000_000;

// Greedy attribute capture (no nested optional whitespace) — the [^>]*? body
// already handles all whitespace internally, so dropping the surrounding \s*
// removes the ambiguity that triggers polynomial backtracking on hostile input.
const TAG_RE = /<\s*(\/)?\s*([a-zA-Z][a-zA-Z0-9]*)([^>]*?)(\/)?>/g;
const ALT_RE = /\balt\s*=\s*"([^"]*)"/i;

const DROP_TAGS = new Set(['actions']);

// Single shared DOMParser. Parsing as text/html lets us delegate ALL XML and
// numeric character entity decoding to the browser — including `&#9731;`,
// `&#x2603;`, `&hearts;`, etc. Manual maps would silently miss numerics and
// every named entity outside the obvious five.
const _parser = typeof DOMParser === 'undefined' ? null : new DOMParser();

function _fallbackDecode(text) {
  return text
    .replaceAll('&amp;', '&').replaceAll('&lt;', '<').replaceAll('&gt;', '>')
    .replaceAll('&quot;', '"').replaceAll('&#39;', "'");
}

function decodeEntities(text) {
  if (!text.includes('&')) return text;
  if (_parser) {
    try {
      // Wrap in <body> so leading whitespace is preserved verbatim.
      const doc = _parser.parseFromString(`<!doctype html><body>${text}`, 'text/html');
      return doc.body.textContent || '';
    } catch (e) {
      console.warn('[markup-extract] DOMParser failed, using fallback:', e);
    }
  }
  // Fallback (no DOM available — Node test env). Covers the common five.
  return _fallbackDecode(text);
}

function _processTag(m, out, skipDepth, pos) {
  const isClose = !!m[1];
  const name = m[2].toLowerCase();
  const isVoid = !!m[4];
  let depth = skipDepth;
  if (DROP_TAGS.has(name)) {
    if (isClose) depth = Math.max(0, depth - 1);
    else if (!isVoid) depth += 1;
  } else if (depth === 0 && name === 'img' && isVoid) {
    const altMatch = ALT_RE.exec(m[3]);
    if (altMatch?.[1]) out.push(altMatch[1]);
  }
  return depth;
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
    skipDepth = _processTag(m, out, skipDepth, pos);
    pos = m.index + m[0].length;
  }
  if (skipDepth === 0 && pos < content.length) {
    out.push(content.slice(pos));
  }
  return decodeEntities(out.join(' '))
    .replaceAll(/\s+/g, ' ')
    .trim();
}
