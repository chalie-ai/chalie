// Renderer for Chalie chat content.
//
// Backend ``services.markup.sanitize()`` (nh3) is the single chokepoint:
// every assistant response is stripped of disallowed tags / attributes
// before it reaches the frontend. This file therefore does NO sanitisation
// — it trusts the backend, renders via innerHTML, then walks the resulting
// tree to:
//   1. Auto-linkify plain-text URLs (the LLM is told never to emit <a>).
//   2. Wire programmatic behaviours on harness-emitted <actions>, <action>,
//      <img> elements (click handlers, lazy loading, allowlist for img src).
//
// All other tags (b, i, u, h1, code, p, ul, li) render as-is.

const URL_RE = /\bhttps?:\/\/[^\s<]+[^\s<.,;:!?)]/g;
const HTTP_PROTOCOLS = new Set(['http:', 'https:']);

function _linkifyTextNode(textNode) {
  const text = textNode.nodeValue;
  if (!text || !URL_RE.test(text)) {
    URL_RE.lastIndex = 0;
    return;
  }
  URL_RE.lastIndex = 0;

  const fragment = document.createDocumentFragment();
  let cursor = 0;
  let match;
  while ((match = URL_RE.exec(text)) !== null) {
    if (match.index > cursor) {
      fragment.appendChild(document.createTextNode(text.slice(cursor, match.index)));
    }
    const anchor = document.createElement('a');
    // Property assignment routes through the browser's URL parser; the
    // protocol gate below relies on the parsed result.
    anchor.href = match[0];
    if (HTTP_PROTOCOLS.has(anchor.protocol)) {
      anchor.target = '_blank';
      anchor.rel = 'noopener noreferrer';
      anchor.textContent = match[0];
      fragment.appendChild(anchor);
    } else {
      // Non-http(s) URL — render the literal text, no link.
      fragment.appendChild(document.createTextNode(match[0]));
    }
    cursor = match.index + match[0].length;
  }
  if (cursor < text.length) {
    fragment.appendChild(document.createTextNode(text.slice(cursor)));
  }
  textNode.parentNode.replaceChild(fragment, textNode);
}

function _linkifyTextNodesIn(root) {
  // Skip <code> subtrees and existing anchors so we never linkify URLs that
  // the user-content has explicitly carved out.
  const walker = document.createTreeWalker(
    root,
    NodeFilter.SHOW_TEXT,
    {
      acceptNode(node) {
        let parent = node.parentNode;
        while (parent && parent !== root) {
          const tag = parent.tagName ? parent.tagName.toLowerCase() : '';
          if (tag === 'code' || tag === 'a') return NodeFilter.FILTER_REJECT;
          parent = parent.parentNode;
        }
        return NodeFilter.FILTER_ACCEPT;
      },
    },
  );
  const targets = [];
  let node;
  while ((node = walker.nextNode()) !== null) {
    targets.push(node);
  }
  for (const t of targets) _linkifyTextNode(t);
}

function _wireImage(img) {
  // Belt-and-braces protocol gate: backend nh3 already restricted img src
  // to http(s), so this is mostly a no-op.
  const probe = document.createElement('a');
  probe.href = img.getAttribute('src') || '';
  if (!HTTP_PROTOCOLS.has(probe.protocol)) {
    img.remove();
    return;
  }
  img.loading = 'lazy';
}

function _wireActionsContainer(container) {
  container.classList.add('chalie-actions-row');
}

function _wireActionButton(actionEl) {
  actionEl.classList.add('chalie-action-button');
  const style = actionEl.getAttribute('style');
  if (style === 'secondary') actionEl.classList.add('chalie-action-button--secondary');
  if (style === 'danger') actionEl.classList.add('chalie-action-button--danger');

  // Move display attrs to dataset for CSS / JS hooks. Overlay-action attrs
  // (execute / collect / target / open-url / payload) are propagated as
  // data-* so AppsPanel._wireOverlayActions can pick them up.
  const label = actionEl.getAttribute('label') || '';
  const value = actionEl.getAttribute('value') || '';
  actionEl.dataset.value = value;
  for (const name of ['execute', 'collect', 'target', 'open-url', 'payload']) {
    const v = actionEl.getAttribute(name);
    if (v !== null) {
      actionEl.dataset[_dashToCamel(name)] = v;
    }
  }
  actionEl.textContent = label;

  // Strip the inline attributes now that we have copies on dataset / classlist.
  for (const name of ['label', 'value', 'execute', 'collect', 'target', 'open-url', 'payload', 'style']) {
    actionEl.removeAttribute(name);
  }

  // Chat-action click handler — overlay actions (execute=...) are wired by
  // AppsPanel._wireOverlayActions, so we skip dispatch in that case to
  // avoid double-firing.
  if (actionEl.dataset.execute) return;
  actionEl.addEventListener('click', () => {
    const row = actionEl.closest('.chalie-actions-row');
    if (row) {
      for (const b of row.querySelectorAll('.chalie-action-button')) {
        b.setAttribute('disabled', '');
      }
    } else {
      actionEl.setAttribute('disabled', '');
    }
    actionEl.classList.add('chalie-action-button--selected');
    document.dispatchEvent(new CustomEvent('chalie:action', {
      detail: { payload: { value, label } },
    }));
  });
}

function _dashToCamel(s) {
  return s.replace(/-([a-z])/g, (_m, c) => c.toUpperCase());
}

function _wireProgrammatic(root) {
  for (const img of root.querySelectorAll('img')) _wireImage(img);
  for (const container of root.querySelectorAll('actions')) _wireActionsContainer(container);
  for (const button of root.querySelectorAll('action')) _wireActionButton(button);
  // <code> styling
  for (const c of root.querySelectorAll('code')) c.classList.add('chalie-code');
}

export function renderMarkupTo(container, content) {
  container.innerHTML = '';
  if (!content) return;
  // Trust the backend chokepoint (services.markup.sanitize). All tags /
  // attributes outside the allowlist are already stripped.
  container.innerHTML = content;
  _linkifyTextNodesIn(container);
  _wireProgrammatic(container);
}

export function renderMarkup(content) {
  const container = document.createElement('div');
  container.classList.add('chalie-markup');
  renderMarkupTo(container, content);
  return container;
}
