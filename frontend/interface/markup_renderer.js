// Renderer for Chalie chat content.
//
// Backend ``services.markup.sanitize()`` (nh3) is the single chokepoint:
// every assistant response is stripped of disallowed tags / attributes
// before it reaches the frontend. This file therefore does NO sanitisation
// — it trusts the backend, renders via innerHTML, then walks the resulting
// tree to:
//   1. Auto-linkify plain-text URLs / emails (the LLM is told never to emit <a>).
//   2. Wire programmatic behaviours on harness-emitted <actions>, <action>,
//      <img> elements (click handlers, lazy loading, scheme gate for img src).
//
// All other tags (b, i, u, h1, code, p, ul, li) render as-is.

import { find as findLinks } from './vendor/linkify.es.mjs';

const HTTP_PROTOCOLS = new Set(['http:', 'https:']);

function _renderLinkAnchor(href, label) {
  const a = document.createElement('a');
  a.href = href;
  // Belt-and-braces: linkify-it only matches http(s)/mailto/etc.; protocol
  // gate here is redundant for those schemes but ensures any future custom
  // schemes can't slip through without an explicit allowlist update.
  if (!HTTP_PROTOCOLS.has(a.protocol) && a.protocol !== 'mailto:') {
    return document.createTextNode(label);
  }
  a.textContent = label;
  if (HTTP_PROTOCOLS.has(a.protocol)) {
    a.target = '_blank';
    a.rel = 'noopener noreferrer';
  }
  return a;
}

function _linkifyTextNode(textNode) {
  const text = textNode.nodeValue;
  if (!text) return;
  const matches = findLinks(text);
  if (!matches.length) return;

  const fragment = document.createDocumentFragment();
  let cursor = 0;
  for (const m of matches) {
    if (m.start > cursor) {
      fragment.appendChild(document.createTextNode(text.slice(cursor, m.start)));
    }
    fragment.appendChild(_renderLinkAnchor(m.href, m.value));
    cursor = m.end;
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

// Inline action attributes that get mirrored onto the live element as ``data-*``.
const _ACTION_DATA_ATTRS = ['execute', 'collect', 'target', 'open-url', 'payload'];

function _wireActionButton(actionEl) {
  actionEl.classList.add('chalie-action-button');
  const style = actionEl.getAttribute('style');
  if (style === 'secondary') actionEl.classList.add('chalie-action-button--secondary');
  if (style === 'danger') actionEl.classList.add('chalie-action-button--danger');

  const label = actionEl.getAttribute('label') || '';
  const value = actionEl.getAttribute('value') || '';

  // Set data-* attributes directly. The DOM handles dash↔camelCase
  // conversion when later read via ``.dataset.openUrl``.
  actionEl.dataset.value = value;
  for (const name of _ACTION_DATA_ATTRS) {
    const v = actionEl.getAttribute(name);
    if (v !== null) actionEl.setAttribute(`data-${name}`, v);
  }
  actionEl.textContent = label;

  // Strip the inline source attributes now that copies live on data-*.
  for (const name of ['label', 'value', ..._ACTION_DATA_ATTRS, 'style']) {
    actionEl.removeAttribute(name);
  }

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
