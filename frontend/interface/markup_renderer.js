// XML markup renderer for Chalie chat content.
//
// Allowlist (11 tags): b, i, u, h1, code, p, ul, li, a, img, actions, action
// Tolerant parser: unknown tags rendered as escaped text. Unclosed tags
// auto-close at EOF. Nesting is rendered as-given; model is trusted.

const LLM_TAGS = new Set(['b', 'i', 'u', 'h1', 'code', 'p', 'ul', 'li', 'a']);
const PROGRAMMATIC_TAGS = new Set(['img', 'actions', 'action']);
const ALLOWED_TAGS = new Set([...LLM_TAGS, ...PROGRAMMATIC_TAGS]);
const VOID_TAGS = new Set(['img', 'action']);

const TAG_RE = /<\s*(\/)?\s*([a-zA-Z][a-zA-Z0-9]*)\s*((?:[a-zA-Z][a-zA-Z0-9-]*\s*=\s*"[^"]*"\s*)*)\s*(\/)?\s*>/g;
const ATTR_RE = /([a-zA-Z][a-zA-Z0-9-]*)\s*=\s*"([^"]*)"/g;

const ENTITY_MAP = {
  '&amp;': '&', '&lt;': '<', '&gt;': '>', '&quot;': '"', '&#39;': "'",
};

function decodeEntities(text) {
  return text.replace(/&(amp|lt|gt|quot|#39);/g, (m) => ENTITY_MAP[m] || m);
}

function tokenize(content) {
  if (!content) return [];
  // Strip NUL bytes — DOM createTextNode silently drops them, leaving an
  // ambiguous gap. Defensive: shouldn't appear in well-formed content.
  if (content.indexOf('\x00') !== -1) {
    content = content.replace(/\x00/g, '');
  }
  const tokens = [];
  let pos = 0;
  TAG_RE.lastIndex = 0;
  let match;
  while ((match = TAG_RE.exec(content)) !== null) {
    if (match.index > pos) {
      tokens.push({ kind: 'text', name: decodeEntities(content.slice(pos, match.index)) });
    }
    const isClose = !!match[1];
    const name = match[2].toLowerCase();
    const attrBlob = match[3] || '';
    const isVoid = !!match[4];
    if (!ALLOWED_TAGS.has(name)) {
      tokens.push({ kind: 'text', name: match[0] });
    } else {
      const attrs = {};
      let am;
      ATTR_RE.lastIndex = 0;
      while ((am = ATTR_RE.exec(attrBlob)) !== null) {
        attrs[am[1].toLowerCase()] = decodeEntities(am[2]);
      }
      if (isClose) {
        tokens.push({ kind: 'close', name });
      } else if (isVoid || VOID_TAGS.has(name)) {
        tokens.push({ kind: 'void', name, attrs });
      } else {
        tokens.push({ kind: 'open', name, attrs });
      }
    }
    pos = match.index + match[0].length;
  }
  if (pos < content.length) {
    tokens.push({ kind: 'text', name: decodeEntities(content.slice(pos)) });
  }
  return tokens;
}

function createElement(name, attrs) {
  const el = document.createElement(name);
  if (name === 'a') {
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
  } else if (name === 'img') {
    const src = attrs.src || '';
    // Same scheme-relative guard as <a>. Note: `data:` URIs are blocked by
    // virtue of not matching this allowlist (intentional — no inline data).
    if (/^(https?:\/\/|\/(?!\/))/i.test(src)) {
      el.setAttribute('src', src);
      el.setAttribute('alt', attrs.alt || '');
      el.setAttribute('loading', 'lazy');
    }
  } else if (name === 'action') {
    el.classList.add('chalie-action-button');
    // Chat actions (LLM-emitted): label + value → dispatch chalie:action.
    // Overlay actions (apps_panel daemon UI): execute/collect/target/open-url/
    // payload/style → propagated as data-* attrs; click wired by host.
    el.setAttribute('data-value', attrs.value || '');
    if (attrs.execute) el.setAttribute('data-execute', attrs.execute);
    if (attrs.collect) el.setAttribute('data-collect', attrs.collect);
    if (attrs.target) el.setAttribute('data-target', attrs.target);
    if (attrs['open-url']) el.setAttribute('data-open-url', attrs['open-url']);
    if (attrs.payload) el.setAttribute('data-payload', attrs.payload);
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
    return el; // action is its own thing — don't append children
  } else if (name === 'actions') {
    el.classList.add('chalie-actions-row');
  } else if (name === 'code') {
    el.classList.add('chalie-code');
  }
  return el;
}

export function renderMarkupTo(container, content) {
  container.innerHTML = '';
  if (!content) return;
  const tokens = tokenize(content);
  const stack = [container];
  for (const tok of tokens) {
    const parent = stack[stack.length - 1];
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
      let stackIdx = -1;
      for (let i = stack.length - 1; i >= 1; i -= 1) {
        if (stack[i].tagName.toLowerCase() === tok.name) {
          stackIdx = i;
          break;
        }
      }
      if (stackIdx > 0) {
        stack.length = stackIdx;
      }
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
