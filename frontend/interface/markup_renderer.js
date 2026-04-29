// XML markup renderer for Chalie chat content.
//
// Allowlist (11 tags): b, i, u, h1, code, p, ul, li, a, img, actions, action
// Tolerant: parsing is delegated to the browser's HTML parser via <template>.
// Unknown tags collapse to their text content. Unclosed tags are auto-closed
// by the parser. Self-closing void tags work natively for `<img/>`; `<action/>`
// is the one allowlisted void that the HTML parser doesn't recognise, so we
// expand it to a balanced pair before parsing.

const ALLOWED_TAGS = new Set(['b', 'i', 'u', 'h1', 'code', 'p', 'ul', 'li', 'a', 'img', 'actions', 'action']);
const VOID_TAGS = new Set(['img', 'action']);

// Bound input to keep parse memory + time predictable on hostile input.
// Real content is bounded by LLM token limits; 5 MB is generous.
const MAX_CONTENT_LEN = 5_000_000;

// HTML5 doesn't recognise <action/> as a void element, so the self-closing
// form leaves it open. Expand to an explicit pair before handing to the
// parser. Linear index scan instead of a regex to keep the pre-processor
// ReDoS-clean.
function _expandActionVoidTags(content) {
  let result = '';
  let i = 0;
  while (i < content.length) {
    const open = content.indexOf('<action', i);
    if (open === -1) {
      result += content.slice(i);
      break;
    }
    // Char following `<action` must be whitespace, `/`, or `>` — anything
    // else (e.g. `<actions>`) is a different tag and gets skipped.
    const after = content[open + 7] || '';
    if (after !== ' ' && after !== '\t' && after !== '\n' && after !== '/' && after !== '>') {
      result += content.slice(i, open + 7);
      i = open + 7;
      continue;
    }
    const close = content.indexOf('>', open);
    if (close === -1) {
      result += content.slice(i);
      break;
    }
    const tag = content.slice(open, close + 1);
    if (tag.endsWith('/>')) {
      result += content.slice(i, open) + tag.slice(0, -2) + '></action>';
    } else {
      result += content.slice(i, close + 1);
    }
    i = close + 1;
  }
  return result;
}

function _normalizeContent(content) {
  let out = content;
  // Strip NUL bytes — DOM createTextNode silently drops them, leaving an
  // ambiguous gap. Defensive: shouldn't appear in well-formed content.
  if (out.includes('\u0000')) {
    out = out.split('\u0000').join('');
  }
  if (out.length > MAX_CONTENT_LEN) {
    out = out.slice(0, MAX_CONTENT_LEN);
  }
  return _expandActionVoidTags(out);
}

function _attrsOf(srcEl) {
  const attrs = {};
  for (const attr of srcEl.attributes) {
    attrs[attr.name.toLowerCase()] = attr.value;
  }
  return attrs;
}

// Allowed link schemes for anchors. ``mailto:`` is included for completeness;
// the URL constructor handles parsing + canonicalisation that CodeQL recognises
// as a proper sanitiser, breaking the DOM-text-to-HTML taint flow that bare
// regex matching does not.
const _ANCHOR_PROTOCOLS = new Set(['http:', 'https:', 'mailto:']);
const _IMAGE_PROTOCOLS = new Set(['http:', 'https:']);

function _isRootRelative(value) {
  // Allow `/path` but reject scheme-relative `//evil.com`.
  return value.startsWith('/') && !value.startsWith('//');
}

function _safeUrl(value, allowedProtocols) {
  if (!value) return null;
  if (_isRootRelative(value)) return value;
  let parsed;
  try {
    parsed = new URL(value);
  } catch (_e) {
    return null;
  }
  if (!allowedProtocols.has(parsed.protocol)) return null;
  return parsed.href;
}

function _wireAnchor(el, attrs) {
  const href = _safeUrl(attrs.href || '#', _ANCHOR_PROTOCOLS);
  if (href === null) return;
  el.setAttribute('href', href);
  if (href.startsWith('http://') || href.startsWith('https://')) {
    el.setAttribute('target', '_blank');
    el.setAttribute('rel', 'noopener noreferrer');
  }
}

function _wireImage(el, attrs) {
  // ``data:`` URIs are blocked by virtue of not matching this allowlist
  // (intentional — no inline data).
  const src = _safeUrl(attrs.src || '', _IMAGE_PROTOCOLS);
  if (src === null) return;
  el.setAttribute('src', src);
  el.setAttribute('alt', attrs.alt || '');
  el.setAttribute('loading', 'lazy');
}

function _wireAction(el, attrs) {
  el.classList.add('chalie-action-button');
  // Chat actions (LLM-emitted): label + value → dispatch chalie:action.
  // Overlay actions (apps_panel daemon UI): execute/collect/target/open-url/
  // payload/style → propagated as data-* attrs; click wired by host.
  el.dataset.value = attrs.value || '';
  if (attrs.execute) el.dataset.execute = attrs.execute;
  if (attrs.collect) el.dataset.collect = attrs.collect;
  if (attrs.target) el.dataset.target = attrs.target;
  if (attrs['open-url']) el.dataset.openUrl = attrs['open-url'];
  if (attrs.payload) el.dataset.payload = attrs.payload;
  if (attrs.style === 'secondary') el.classList.add('chalie-action-button--secondary');
  if (attrs.style === 'danger') el.classList.add('chalie-action-button--danger');
  el.textContent = attrs.label || '';
  // Chat-action click handler. Overlay actions have data-execute and are
  // wired by AppsPanel._wireOverlayActions; we skip dispatch in that case
  // to avoid double-firing.
  if (!attrs.execute) {
    el.addEventListener('click', () => {
      // One-time use: disable siblings in the same <actions> row.
      const row = el.closest('.chalie-actions-row');
      if (row) {
        for (const b of row.querySelectorAll('.chalie-action-button')) {
          b.setAttribute('disabled', '');
        }
      } else {
        el.setAttribute('disabled', '');
      }
      el.classList.add('chalie-action-button--selected');
      document.dispatchEvent(new CustomEvent('chalie:action', {
        detail: {
          payload: { value: attrs.value || '', label: attrs.label || '' },
        },
      }));
    });
  }
}

function _createElement(name, attrs) {
  const el = document.createElement(name);
  if (name === 'a') {
    _wireAnchor(el, attrs);
  } else if (name === 'img') {
    _wireImage(el, attrs);
  } else if (name === 'action') {
    _wireAction(el, attrs);
  } else if (name === 'actions') {
    el.classList.add('chalie-actions-row');
  } else if (name === 'code') {
    el.classList.add('chalie-code');
  }
  return el;
}

function _appendNode(parent, srcNode) {
  if (srcNode.nodeType === Node.TEXT_NODE) {
    parent.appendChild(document.createTextNode(srcNode.nodeValue));
    return;
  }
  if (srcNode.nodeType !== Node.ELEMENT_NODE) {
    return; // comments, processing instructions, etc. are dropped silently
  }
  const tag = srcNode.tagName.toLowerCase();
  if (!ALLOWED_TAGS.has(tag)) {
    // Unknown tags collapse to their visible text content — preserves human
    // intent without leaking unallowlisted markup into the DOM.
    parent.appendChild(document.createTextNode(srcNode.textContent || ''));
    return;
  }
  const el = _createElement(tag, _attrsOf(srcNode));
  parent.appendChild(el);
  // Void tags + the action button (whose textContent is set from `label`)
  // do not recurse into source children.
  if (VOID_TAGS.has(tag)) return;
  for (const child of srcNode.childNodes) {
    _appendNode(el, child);
  }
}

function _parseToFragment(content) {
  const template = document.createElement('template');
  template.innerHTML = content;
  return template.content;
}

export function renderMarkupTo(container, content) {
  container.innerHTML = '';
  if (!content) return;
  const fragment = _parseToFragment(_normalizeContent(content));
  for (const child of fragment.childNodes) {
    _appendNode(container, child);
  }
}

export function renderMarkup(content) {
  const container = document.createElement('div');
  container.classList.add('chalie-markup');
  renderMarkupTo(container, content);
  return container;
}
