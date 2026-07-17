/**
 * Markup rendering and text extraction for Chalie chat content.
 *
 * Security model: the backend nh3 sanitize() is the single chokepoint — every
 * response is stripped of disallowed tags/attrs before reaching the frontend,
 * so this composable does NOT re-sanitize. It parses through a detached DOM
 * node (not raw string concat), then linkifies text, lazy-loads (protocol-gated)
 * <img>, and adds CSS classes. Because parsing goes through the DOM, only
 * nh3-allowed elements plus the safe wrappers added here reach the v-html
 * output. extractText strips all tags for TTS.
 *
 * The img/link gate uses a base-less ``new URL()`` parse (not an anchor-probe).
 * Base-less ``new URL()`` throws on relative, scheme-less, and protocol-relative
 * input, so those are refused — unlike ``a.href = url; a.protocol`` which
 * resolves them against the page origin and would wrongly pass them through.
 */

import { find as findLinks } from '../vendor/linkify.es.mjs';
import { isAbsoluteHttpUrl } from '../utils/url';

function _renderLinkAnchor(href: string, label: string): Node {
  const a = document.createElement('a');
  a.href = href;
  const isMailto = href.trim().toLowerCase().startsWith('mailto:');
  // Belt-and-braces protocol gate — linkify only matches http(s)/mailto/etc.
  if (!isAbsoluteHttpUrl(href) && !isMailto) {
    return document.createTextNode(label);
  }
  a.textContent = label;
  if (isAbsoluteHttpUrl(href)) {
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

function _wireImage(img: HTMLImageElement): void {
  if (!isAbsoluteHttpUrl(img.getAttribute('src') ?? '')) {
    img.remove();
    return;
  }
  img.loading = 'lazy';
}

function _wireProgrammatic(root: Element): void {
  for (const img of root.querySelectorAll('img')) _wireImage(img as HTMLImageElement);
  for (const c of root.querySelectorAll('code')) c.classList.add('chalie-code');
}

/** Process backend-sanitized markup into a safe HTML string for v-html. */
export function renderMarkup(content: string): string {
  if (!content) return '';
  const container = document.createElement('div');
  container.className = 'chalie-markup';
  container.innerHTML = content;
  _linkifyTextNodesIn(container);
  _wireProgrammatic(container);
  return container.innerHTML;
}

function _walk(node: Node, out: string[]): void {
  if (node.nodeType === Node.TEXT_NODE) {
    if (node.nodeValue) out.push(node.nodeValue);
    return;
  }
  if (node.nodeType !== Node.ELEMENT_NODE) return;

  const el = node as Element;
  const tag = el.tagName.toLowerCase();
  if (tag === 'img') {
    const alt = (el.getAttribute('alt') ?? '').trim();
    if (alt) out.push(alt);
    return;
  }
  for (const child of el.childNodes) {
    _walk(child, out);
  }
}

/** Strip tags to plain text for TTS. */
export function extractText(content: string): string {
  if (!content) return '';
  const template = document.createElement('template');
  template.innerHTML = content;
  const out: string[] = [];
  for (const child of template.content.childNodes) {
    _walk(child, out);
  }
  return out.join(' ').replaceAll(/\s+/g, ' ').trim();
}
