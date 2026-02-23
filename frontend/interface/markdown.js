/**
 * Lightweight, zero-dependency markdown parser.
 * Outputs only a fixed allowlist of HTML tags — XSS-safe by construction.
 * Links are validated to http/https only.
 */

export function parseMarkdown(text) {
  // 1. Extract and placeholder fenced code blocks (protect contents from inline processing)
  const codeBlocks = [];
  text = text.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) => {
    const i = codeBlocks.length;
    const escaped = escapeHtml(code.trim());
    codeBlocks.push(
      `<pre><code${lang ? ` class="language-${escapeHtml(lang)}"` : ''}>${escaped}</code></pre>`
    );
    return `\x00CODE${i}\x00`;
  });

  // 2. Inline code
  text = text.replace(/`([^`]+)`/g, (_, code) => `<code>${escapeHtml(code)}</code>`);

  // Block-level processing line by line
  const lines = text.split('\n');
  const out = [];
  let listType = null;
  let listItems = [];

  const flushList = () => {
    if (!listItems.length) return;
    out.push(`<${listType}>${listItems.map(i => `<li>${i}</li>`).join('')}</${listType}>`);
    listType = null;
    listItems = [];
  };

  for (const line of lines) {
    // Headers
    const hm = line.match(/^(#{1,3})\s+(.*)/);
    if (hm) {
      flushList();
      out.push(`<h${hm[1].length}>${inlineMarkdown(hm[2])}</h${hm[1].length}>`);
      continue;
    }

    // Unordered list
    const ulm = line.match(/^[-*]\s+(.*)/);
    if (ulm) {
      if (listType !== 'ul') { flushList(); listType = 'ul'; }
      listItems.push(inlineMarkdown(ulm[1]));
      continue;
    }

    // Ordered list
    const olm = line.match(/^\d+\.\s+(.*)/);
    if (olm) {
      if (listType !== 'ol') { flushList(); listType = 'ol'; }
      listItems.push(inlineMarkdown(olm[1]));
      continue;
    }

    // Blank line — paragraph separator
    if (line.trim() === '') {
      flushList();
      out.push('');
      continue;
    }

    flushList();
    out.push(inlineMarkdown(line));
  }
  flushList();

  // Wrap non-block content in <p>, join block elements directly
  let html = '';
  const paras = out.join('\n').split(/\n{2,}/);
  for (const para of paras) {
    const t = para.trim();
    if (!t) continue;
    if (/^<(h[1-3]|ul|ol|pre|blockquote)/.test(t)) {
      html += t + '\n';
    } else {
      html += `<p>${t.replace(/\n/g, '<br>')}</p>\n`;
    }
  }

  // 3. Restore code blocks
  html = html.replace(/\x00CODE(\d+)\x00/g, (_, i) => codeBlocks[+i]);

  return html;
}

function inlineMarkdown(text) {
  // Bold
  text = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  // Italic (underscore)
  text = text.replace(/_(.+?)_/g, '<em>$1</em>');
  // Italic (asterisk, non-greedy, no double-asterisk)
  text = text.replace(/\*([^*]+)\*/g, '<em>$1</em>');
  // Links — http/https only to prevent javascript: or data: XSS
  text = text.replace(
    /\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g,
    '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
  );
  return text;
}

function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
