/**
 * Markup rendering and text extraction for Chalie chat content.
 *
 * Security model (preserved from markup_renderer.js):
 *   The backend services.markup.sanitize() (nh3) is the single chokepoint:
 *   every assistant response is stripped of disallowed tags/attributes before
 *   it reaches the frontend.  This composable therefore does NOT re-sanitize —
 *   it trusts the backend, renders via a detached DOM node, then:
 *     1. Auto-linkifies plain-text URLs/emails in text nodes.
 *     2. Wires <action> buttons to emit the 'chalie:action' CustomEvent.
 *     3. Adds lazy-loading to <img> tags (protocol-gated).
 *     4. Adds CSS classes to <code>, <actions>, <action> elements.
 *
 *   The result is returned as an HTML string for use with v-html.  Because the
 *   content is first parsed through the detached DOM (not concatenated as a raw
 *   string), the only element types that reach v-html output are those the
 *   backend's nh3 allowlist permits plus the safe wrappers added here
 *   (<a>, class attributes, data-* attributes, disabled attribute on buttons).
 *
 * extractText:
 *   Strips all tags for TTS. Ported from markup_extract.js.
 */

import { find as findLinks } from '../vendor/linkify.es.mjs';
import { emit } from './useEventBus';

const HTTP_PROTOCOLS = new Set(['http:', 'https:']);

// ── Linkifier ─────────────────────────────────────────────────────────────

function _renderLinkAnchor(href: string, label: string): Node {
  const a = document.createElement('a');
  a.href = href;
  // Belt-and-braces protocol gate — linkify only matches http(s)/mailto/etc.
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

function _linkifyTextNode(textNode: Text): void {
  const text = textNode.nodeValue;
  if (!text) return;
  const matches = findLinks(text) as Array<{ start: number; end: number; href: string; value: string }>;
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
  textNode.parentNode?.replaceChild(fragment, textNode);
}

function _linkifyTextNodesIn(root: Element): void {
  // Skip <code> subtrees and existing anchors so we never linkify URLs that
  // the content has explicitly carved out.
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      let parent: Node | null = node.parentNode;
      while (parent && parent !== root) {
        const tag = (parent as Element).tagName?.toLowerCase() ?? '';
        if (tag === 'code' || tag === 'a') return NodeFilter.FILTER_REJECT;
        parent = parent.parentNode;
      }
      return NodeFilter.FILTER_ACCEPT;
    },
  });

  const targets: Text[] = [];
  let node: Node | null;
  while ((node = walker.nextNode()) !== null) {
    targets.push(node as Text);
  }
  for (const t of targets) _linkifyTextNode(t);
}

// ── Image wiring ──────────────────────────────────────────────────────────

function _wireImage(img: HTMLImageElement): void {
  const probe = document.createElement('a');
  probe.href = img.getAttribute('src') ?? '';
  if (!HTTP_PROTOCOLS.has(probe.protocol)) {
    img.remove();
    return;
  }
  img.loading = 'lazy';
}

// ── Action button wiring ───────────────────────────────────────────────────

const _ACTION_DATA_ATTRS = ['execute', 'collect', 'target', 'open-url', 'payload'] as const;

function _wireActionsContainer(container: Element): void {
  container.classList.add('chalie-actions-row');
}

function _wireActionButton(actionEl: Element): void {
  actionEl.classList.add('chalie-action-button');
  const style = actionEl.getAttribute('style');
  if (style === 'secondary') actionEl.classList.add('chalie-action-button--secondary');
  if (style === 'danger') actionEl.classList.add('chalie-action-button--danger');

  const label = actionEl.getAttribute('label') ?? '';
  const value = actionEl.getAttribute('value') ?? '';

  (actionEl as HTMLElement).dataset['value'] = value;
  for (const name of _ACTION_DATA_ATTRS) {
    const v = actionEl.getAttribute(name);
    if (v !== null) actionEl.setAttribute(`data-${name}`, v);
  }
  actionEl.textContent = label;

  for (const name of ['label', 'value', ..._ACTION_DATA_ATTRS, 'style'] as const) {
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
    emit('chalie:action', { payload: { value, label } });
  });
}

function _wireProgrammatic(root: Element): void {
  for (const img of root.querySelectorAll('img')) _wireImage(img as HTMLImageElement);
  for (const container of root.querySelectorAll('actions')) _wireActionsContainer(container);
  for (const button of root.querySelectorAll('action')) _wireActionButton(button);
  for (const c of root.querySelectorAll('code')) c.classList.add('chalie-code');
}

// ── Public API ─────────────────────────────────────────────────────────────

/**
 * Process backend-sanitized markup and return a safe HTML string for v-html.
 *
 * The returned string is produced by DOM manipulation (not string concatenation),
 * so it is safe to use with v-html on the assumption that the backend's nh3
 * sanitizer has already stripped disallowed tags/attributes.
 */
export function renderMarkup(content: string): string {
  if (!content) return '';
  const container = document.createElement('div');
  container.className = 'chalie-markup';
  container.innerHTML = content;
  _linkifyTextNodesIn(container);
  _wireProgrammatic(container);
  return container.innerHTML;
}

// ── Plain-text extraction (for TTS) ───────────────────────────────────────
// Ported from markup_extract.js

const _DROP_TAGS = new Set(['actions']);

function _walk(node: Node, out: string[]): void {
  if (node.nodeType === Node.TEXT_NODE) {
    if (node.nodeValue) out.push(node.nodeValue);
    return;
  }
  if (node.nodeType !== Node.ELEMENT_NODE) return;

  const el = node as Element;
  const tag = el.tagName.toLowerCase();
  if (_DROP_TAGS.has(tag)) return;
  if (tag === 'img') {
    const alt = (el.getAttribute('alt') ?? '').trim();
    if (alt) out.push(alt);
    return;
  }
  for (const child of el.childNodes) {
    _walk(child, out);
  }
}

/**
 * Strip markup tags and return plain text for TTS consumption.
 * Drops <actions> subtrees so button labels don't leak into speech.
 */
export function extractText(content: string): string {
  if (!content) return '';
  const template = document.createElement('template');
  template.innerHTML = content;
  // Drop <actions> subtrees — querySelectorAll returns static NodeList.
  for (const node of template.content.querySelectorAll('actions')) {
    node.remove();
  }
  const out: string[] = [];
  for (const child of template.content.childNodes) {
    _walk(child, out);
  }
  return out.join(' ').replaceAll(/\s+/g, ' ').trim();
}

export function useMarkup() {
  return { renderMarkup, extractText };
}
