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
// parser. Bounded, anchored, no nested quantifiers — same predictable cost
// as a string operation.
const ACTION_SELF_CLOSE_RE = /<action\b([^>]*?)\s*\/>/g;

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
  return out.replaceAll(ACTION_SELF_CLOSE_RE, '<action$1></action>');
}

function _attrsOf(srcEl) {
  const attrs = {};
  for (const attr of srcEl.attributes) {
    attrs[attr.name.toLowerCase()] = attr.value;
  }
  return attrs;
}

function _wireAnchor(el, attrs) {
  const href = attrs.href || '#';
  // Sanitise href: only allow http(s)://, mailto:, or root-relative `/`
  // that is NOT scheme-relative `//evil.com` (the browser would otherwise
  // resolve `//x` to `https://x` — open-redirect / phishing surface).
  if (/^(https?:\/\/|mailto:|\/(?!\/))/i.test(href)) {
    el.setAttribute('href', href);
    if (/^https?:\/\//i.test(href)) {
      el.setAttribute('target', '_blank');
      el.setAttribute('rel', 'noopener noreferrer');
    }
  }
}

function _wireImage(el, attrs) {
  const src = attrs.src || '';
  // Same scheme-relative guard as <a>. Note: `data:` URIs are blocked by
  // virtue of not matching this allowlist (intentional — no inline data).
  if (/^(https?:\/\/|\/(?!\/))/i.test(src)) {
    el.setAttribute('src', src);
    el.setAttribute('alt', attrs.alt || '');
    el.setAttribute('loading', 'lazy');
  }
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
