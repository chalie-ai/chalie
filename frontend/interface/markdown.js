/**
 * Markdown parser — wraps `marked` with XSS-safe configuration.
 * Outputs only safe HTML tags. Links validated to http/https only.
 */
import { marked } from './lib/marked.esm.js';

function escapeHtml(str = '') {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

marked.use({
  gfm: true,
  breaks: true,
  renderer: {
    // Block raw HTML — escape instead of rendering (XSS safety)
    html({ text }) {
      return escapeHtml(text);
    },

    // Links — http/https only, with whitespace-trimmed href validation
    link({ href, title, tokens }) {
      if (!href || !/^https?:\/\//i.test(href.trim())) {
        return this.parser.parseInline(tokens);
      }
      const titleAttr = title ? ` title="${escapeHtml(title)}"` : '';
      return `<a href="${escapeHtml(href)}"${titleAttr} target="_blank" rel="noopener noreferrer">${this.parser.parseInline(tokens)}</a>`;
    },

    // Images — strip entirely (not supported in chat context)
    image({ text }) {
      return escapeHtml(text || '');
    },
  },
});

/**
 * Parse markdown text to safe HTML.
 * @param {string} text — raw markdown from LLM response
 * @returns {string} — sanitized HTML
 */
export function parseMarkdown(text) {
  if (!text) return '';
  return marked.parse(text);
}
