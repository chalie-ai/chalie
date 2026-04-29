// Plaintext extractor for the speak button (TTS).
//
// Backend ``services.markup.sanitize()`` (nh3) is the single chokepoint for
// allowlist enforcement. This file therefore does NO sanitisation; it simply
// strips all tags via a throwaway <template>, drops <actions> subtrees so
// button labels do not leak into the spoken output, and uses <img alt> when
// present.

const DROP_TAGS = new Set(['actions']);

function _stripActionsSubtrees(root) {
  for (const node of [...root.querySelectorAll('actions')]) {
    node.remove();
  }
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
  if (DROP_TAGS.has(tag)) return;
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
  const template = document.createElement('template');
  template.innerHTML = content;
  _stripActionsSubtrees(template.content);
  const out = [];
  for (const child of template.content.childNodes) {
    _walk(child, out);
  }
  return out.join(' ').replaceAll(/\s+/g, ' ').trim();
}
