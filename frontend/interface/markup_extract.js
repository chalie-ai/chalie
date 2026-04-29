// Extract plaintext from XML markup for the speak button (TTS).
// Drops <actions> entirely. Uses <img alt> when present.
//
// Parsing is delegated to the browser's HTML parser via <template>; entity
// decoding, attribute parsing, and nesting are all handled natively. The only
// pre-processing is NUL-byte stripping + bounding the input length.

// Bound input to keep parse memory + time predictable on hostile input.
// Real content is bounded by LLM token limits; 5 MB is generous.
const MAX_CONTENT_LEN = 5_000_000;

const DROP_TAGS = new Set(['actions']);

function _normalizeContent(content) {
  let out = content;
  if (out.includes('\u0000')) {
    out = out.split('\u0000').join('');
  }
  if (out.length > MAX_CONTENT_LEN) {
    out = out.slice(0, MAX_CONTENT_LEN);
  }
  return out;
}

function _parseToFragment(content) {
  const template = document.createElement('template');
  template.innerHTML = content;
  return template.content;
}

function _walk(node, out) {
  if (node.nodeType === Node.TEXT_NODE) {
    out.push(node.nodeValue);
    return;
  }
  if (node.nodeType !== Node.ELEMENT_NODE) {
    return;
  }
  const tag = node.tagName.toLowerCase();
  if (DROP_TAGS.has(tag)) {
    return; // <actions>...</actions> is silenced for TTS
  }
  if (tag === 'img') {
    const alt = (node.getAttribute('alt') || '').trim();
    if (alt) out.push(alt);
    return;
  }
  for (const child of node.childNodes) {
    _walk(child, out);
  }
}

export function extractPlaintext(content) {
  if (!content) return '';
  const fragment = _parseToFragment(_normalizeContent(content));
  const out = [];
  for (const child of fragment.childNodes) {
    _walk(child, out);
  }
  return out.join(' ').replaceAll(/\s+/g, ' ').trim();
}
