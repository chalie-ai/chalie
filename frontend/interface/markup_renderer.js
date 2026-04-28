// XML markup renderer for Chalie chat content.
//
// Allowlist (11 tags): b, i, u, h1, code, p, ul, li, a, img, actions, action
// Tolerant parser: unknown tags rendered as escaped text. Unclosed tags
// auto-close at EOF. Nesting is rendered as-given; model is trusted.

const LLM_TAGS = new Set(['b', 'i', 'u', 'h1', 'code', 'p', 'ul', 'li', 'a']);
const PROGRAMMATIC_TAGS = new Set(['img', 'actions', 'action']);
const ALLOWED_TAGS = new Set([...LLM_TAGS, ...PROGRAMMATIC_TAGS]);
const VOID_TAGS = new Set(['img', 'action']);

// Bound input to avoid pathological regex backtracking on hostile input.
// Real content is bounded by LLM token limits; 5 MB is generous.
const MAX_CONTENT_LEN = 5_000_000;

// Non-backtracking variant: each attribute requires a leading \s+ separator,
// removing the ambiguity between trailing \s* inside the repetition and the
// outer \s* before (\/)?>. Mitigates polynomial-runtime regex hotspot.
// \w used for [a-zA-Z0-9] to reduce regex complexity (underscore is harmless
// since the allowlist governs what tags/attrs are actually processed).
const TAG_RE = /<\s*(\/)?\s*([a-zA-Z]\w*)((?:\s+[a-zA-Z][\w-]*\s*=\s*"[^"]*")*)\s*(\/)?>/g;
const ATTR_RE = /([a-zA-Z][\w-]*)\s*=\s*"([^"]*)"/g;

const ENTITY_MAP = {
  '&amp;': '&', '&lt;': '<', '&gt;': '>', '&quot;': '"', '&#39;': "'",
};

function decodeEntities(text) {
  return text.replaceAll(/&(amp|lt|gt|quot|#39);/g, (m) => ENTITY_MAP[m] || m);
}

function _parseTagToken(match) {
  const isClose = !!match[1];
  const name = match[2].toLowerCase();
  const attrBlob = match[3] || '';
  const isVoid = !!match[4];
  if (ALLOWED_TAGS.has(name)) {
    const attrs = {};
    let am;
    ATTR_RE.lastIndex = 0;
    while ((am = ATTR_RE.exec(attrBlob)) !== null) {
      attrs[am[1].toLowerCase()] = decodeEntities(am[2]);
    }
    if (isClose) return { kind: 'close', name };
    if (isVoid || VOID_TAGS.has(name)) return { kind: 'void', name, attrs };
    return { kind: 'open', name, attrs };
  }
  return { kind: 'text', name: match[0] };
}

function tokenize(content) {
  if (!content) return [];
  // Strip NUL bytes — DOM createTextNode silently drops them, leaving an
  // ambiguous gap. Defensive: shouldn't appear in well-formed content.
  // Use string split/join (no regex) and \u0000 escape (NOT a literal control
  // character) to satisfy lint rules forbidding raw control bytes in source.
  if (content.includes('\u0000')) {
    content = content.split('\u0000').join('');
  }
  if (content.length > MAX_CONTENT_LEN) {
    content = content.slice(0, MAX_CONTENT_LEN);
  }
  const tokens = [];
  let pos = 0;
  TAG_RE.lastIndex = 0;
  let match;
  while ((match = TAG_RE.exec(content)) !== null) {
    if (match.index > pos) {
      tokens.push({ kind: 'text', name: decodeEntities(content.slice(pos, match.index)) });
    }
    tokens.push(_parseTagToken(match));
    pos = match.index + match[0].length;
  }
  if (pos < content.length) {
    tokens.push({ kind: 'text', name: decodeEntities(content.slice(pos)) });
  }
  return tokens;
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

function createElement(name, attrs) {
  const el = document.createElement(name);
  if (name === 'a') {
    _wireAnchor(el, attrs);
  } else if (name === 'img') {
    _wireImage(el, attrs);
  } else if (name === 'action') {
    _wireAction(el, attrs);
    return el; // action is its own thing — don't append children
  } else if (name === 'actions') {
    el.classList.add('chalie-actions-row');
  } else if (name === 'code') {
    el.classList.add('chalie-code');
  }
  return el;
}

function _resolveCloseTag(stack, name) {
  for (let i = stack.length - 1; i >= 1; i -= 1) {
    if (stack[i].tagName.toLowerCase() === name) return i;
  }
  return -1;
}

export function renderMarkupTo(container, content) {
  container.innerHTML = '';
  if (!content) return;
  const tokens = tokenize(content);
  const stack = [container];
  for (const tok of tokens) {
    const parent = stack.at(-1);
    if (tok.kind === 'text') {
      parent.appendChild(document.createTextNode(tok.name));
    } else if (tok.kind === 'open') {
      const el = createElement(tok.name, tok.attrs);
      parent.appendChild(el);
      stack.push(el);
    } else if (tok.kind === 'close') {
      // Auto-close intervening tags if a close arrives out of order.
      // Manual reverse scan (Array.prototype.findLastIndex requires Safari
      // 15.4+ / 2022 — keep this baseline-safe).
      const stackIdx = _resolveCloseTag(stack, tok.name);
      if (stackIdx > 0) stack.length = stackIdx;
    } else if (tok.kind === 'void') {
      const el = createElement(tok.name, tok.attrs);
      parent.appendChild(el);
    }
  }
}

export function renderMarkup(content) {
  const container = document.createElement('div');
  container.classList.add('chalie-markup');
  renderMarkupTo(container, content);
  return container;
}
