/**
 * Block renderer — converts an array of typed block objects into DOM nodes.
 *
 * Replaces the parseMarkdown() → innerHTML pipeline for structured content.
 * Blocks are JSON objects with a required "type" field and type-specific fields.
 *
 * Usage:
 *   import { BlockRenderer } from './blocks.js';
 *   const renderer = new BlockRenderer();
 *   const fragment = renderer.render(blocks);
 *   container.appendChild(fragment);
 */
import { escHtml } from './utils.js';

// ---------------------------------------------------------------------------
// URL safety helper
// ---------------------------------------------------------------------------

/**
 * Return true only for http/https URLs. Rejects data:, javascript:, etc.
 * @param {string} url
 * @returns {boolean}
 */
function isSafeUrl(url) {
  if (typeof url !== 'string') return false;
  return /^https?:\/\//i.test(url.trim());
}

/**
 * Return true for relative URLs (starting with /) or safe absolute URLs.
 * Used for image src where relative paths are acceptable.
 * @param {string} url
 * @returns {boolean}
 */
function isSafeImageUrl(url) {
  if (typeof url !== 'string') return false;
  return url.trim().startsWith('/') || isSafeUrl(url);
}

// ---------------------------------------------------------------------------
// BlockRenderer
// ---------------------------------------------------------------------------

export class BlockRenderer {
  /**
   * Render an array of block objects into a DocumentFragment.
   *
   * @param {Array<Object>} blocks — array of block descriptors
   * @returns {DocumentFragment}
   */
  render(blocks) {
    const fragment = document.createDocumentFragment();

    if (!Array.isArray(blocks) || blocks.length === 0) {
      return fragment;
    }

    for (const block of blocks) {
      if (!block || typeof block !== 'object' || !block.type) continue;

      const el = this._renderBlock(block);
      if (el) fragment.appendChild(el);
    }

    return fragment;
  }

  // ---------------------------------------------------------------------------
  // Block dispatch
  // ---------------------------------------------------------------------------

  /**
   * Dispatch a single block to the appropriate renderer.
   * Returns an Element, or null to skip.
   * @param {Object} block
   * @returns {Element|null}
   */
  _renderBlock(block) {
    switch (block.type) {
      case 'text':      return this._renderText(block);
      case 'header':    return this._renderHeader(block);
      case 'code':      return this._renderCode(block);
      case 'list':      return this._renderList(block);
      case 'table':     return this._renderTable(block);
      case 'keyvalue':  return this._renderKeyValue(block);
      case 'image':     return this._renderImage(block);
      case 'link':      return this._renderLink(block);
      case 'divider':   return this._renderDivider();
      case 'actions':   return this._renderActions(block);
      case 'carousel':  return this._renderCarousel(block);
      default:          return this._renderUnknown(block);
    }
  }

  // ---------------------------------------------------------------------------
  // Individual block renderers
  // ---------------------------------------------------------------------------

  /**
   * text block — plain text or inline-markdown paragraph.
   * { type: "text", content: "...", format: "plain"|"markdown" }
   */
  _renderText(block) {
    const content = block.content;
    if (typeof content !== 'string' || content === '') return null;

    const p = document.createElement('p');
    p.className = 'block-text';

    if (block.format === 'markdown') {
      p.innerHTML = this._parseInlineMarkdown(content);
    } else {
      // Plain text — textContent is XSS-safe
      p.textContent = content;
    }

    return p;
  }

  /**
   * header block — h2–h6 heading (never h1 in chat context).
   * { type: "header", content: "...", level: 1-6 }
   */
  _renderHeader(block) {
    const content = block.content;
    if (typeof content !== 'string' || content === '') return null;

    // Clamp level to 2–6; treat missing or out-of-range as 2
    const raw = parseInt(block.level, 10);
    const level = (!isNaN(raw) && raw >= 1 && raw <= 6) ? Math.max(2, raw) : 2;

    const h = document.createElement(`h${level}`);
    h.className = 'block-header';
    h.textContent = content; // Headers are plain text
    return h;
  }

  /**
   * code block — preformatted code with optional language label.
   * { type: "code", content: "...", language: "python" }
   */
  _renderCode(block) {
    const content = block.content;
    if (typeof content !== 'string') return null;

    const pre = document.createElement('pre');
    pre.className = 'block-code';

    // Optional language label chip above the code
    if (block.language && typeof block.language === 'string') {
      pre.dataset.language = block.language;
      const label = document.createElement('span');
      label.className = 'block-code__lang';
      label.textContent = block.language;
      pre.appendChild(label);
    }

    const code = document.createElement('code');
    // textContent preserves whitespace and prevents HTML interpretation
    code.textContent = content;
    pre.appendChild(code);

    return pre;
  }

  /**
   * list block — unordered or ordered list with inline-markdown items.
   * { type: "list", style: "unordered"|"ordered", items: ["..."] }
   */
  _renderList(block) {
    if (!Array.isArray(block.items) || block.items.length === 0) return null;

    const tag = block.style === 'ordered' ? 'ol' : 'ul';
    const list = document.createElement(tag);
    list.className = 'block-list';

    for (const item of block.items) {
      if (typeof item !== 'string') continue;
      const li = document.createElement('li');
      li.innerHTML = this._parseInlineMarkdown(item);
      list.appendChild(li);
    }

    return list.children.length > 0 ? list : null;
  }

  /**
   * table block — scrollable table with headers and rows.
   * { type: "table", headers: ["Name", ...], rows: [["val", ...], ...] }
   */
  _renderTable(block) {
    const headers = Array.isArray(block.headers) ? block.headers : [];
    const rows = Array.isArray(block.rows) ? block.rows : [];
    if (headers.length === 0 && rows.length === 0) return null;

    const wrap = document.createElement('div');
    wrap.className = 'block-table-wrap';

    const table = document.createElement('table');
    table.className = 'block-table';

    // thead
    if (headers.length > 0) {
      const thead = document.createElement('thead');
      const tr = document.createElement('tr');
      for (const h of headers) {
        const th = document.createElement('th');
        th.textContent = typeof h === 'string' ? h : String(h ?? '');
        tr.appendChild(th);
      }
      thead.appendChild(tr);
      table.appendChild(thead);
    }

    // tbody
    if (rows.length > 0) {
      const tbody = document.createElement('tbody');
      for (const row of rows) {
        if (!Array.isArray(row)) continue;
        const tr = document.createElement('tr');
        for (const cell of row) {
          const td = document.createElement('td');
          td.textContent = typeof cell === 'string' ? cell : String(cell ?? '');
          tr.appendChild(td);
        }
        tbody.appendChild(tr);
      }
      table.appendChild(tbody);
    }

    wrap.appendChild(table);
    return wrap;
  }

  /**
   * keyvalue block — definition list of key/value pairs.
   * { type: "keyvalue", pairs: [{ key: "K", value: "V" }] }
   */
  _renderKeyValue(block) {
    if (!Array.isArray(block.pairs) || block.pairs.length === 0) return null;

    const dl = document.createElement('dl');
    dl.className = 'block-keyvalue';

    for (const pair of block.pairs) {
      if (!pair || typeof pair !== 'object') continue;
      const dt = document.createElement('dt');
      dt.textContent = typeof pair.key === 'string' ? pair.key : String(pair.key ?? '');
      const dd = document.createElement('dd');
      dd.textContent = typeof pair.value === 'string' ? pair.value : String(pair.value ?? '');
      dl.appendChild(dt);
      dl.appendChild(dd);
    }

    return dl.children.length > 0 ? dl : null;
  }

  /**
   * image block — lazy-loaded image.
   * { type: "image", url: "/path/to/img.jpg", alt: "description" }
   */
  _renderImage(block) {
    const url = block.url;
    if (!isSafeImageUrl(url)) return null;

    const img = document.createElement('img');
    img.className = 'block-image';
    img.src = url;
    img.alt = typeof block.alt === 'string' ? block.alt : '';
    img.loading = 'lazy';
    return img;
  }

  /**
   * link block — inline anchor (http/https only).
   * { type: "link", url: "https://...", text: "Click here" }
   */
  _renderLink(block) {
    const url = block.url;
    const text = typeof block.text === 'string' ? block.text : '';

    if (!isSafeUrl(url)) {
      // Degrade to plain text if URL is unsafe or missing
      if (!text) return null;
      const span = document.createElement('span');
      span.className = 'block-link';
      span.textContent = text;
      return span;
    }

    const a = document.createElement('a');
    a.className = 'block-link';
    a.href = url;
    a.textContent = text || url;
    a.target = '_blank';
    a.rel = 'noopener noreferrer';
    return a;
  }

  /**
   * divider block — horizontal rule.
   * { type: "divider" }
   */
  _renderDivider() {
    const hr = document.createElement('hr');
    hr.className = 'block-divider';
    return hr;
  }

  /**
   * actions block — one-time-use action buttons.
   * { type: "actions", buttons: [{ label: "Yes", skill: "memorize", payload: {...} }] }
   */
  _renderActions(block) {
    if (!Array.isArray(block.buttons) || block.buttons.length === 0) return null;

    const container = document.createElement('div');
    container.className = 'block-actions';

    for (const btn of block.buttons) {
      if (!btn || typeof btn !== 'object') continue;
      const label = typeof btn.label === 'string' ? btn.label : '';
      if (!label) continue;

      const button = document.createElement('button');
      button.className = 'block-action-btn';
      button.textContent = label;

      button.addEventListener('click', () => {
        // Disable ALL buttons in this actions block (one-time use)
        for (const b of container.querySelectorAll('.block-action-btn')) {
          b.disabled = true;
        }
        button.classList.add('block-action-btn--selected');

        document.dispatchEvent(new CustomEvent('chalie:action', {
          detail: { payload: btn.payload ?? null },
        }));
      });

      container.appendChild(button);
    }

    return container.children.length > 0 ? container : null;
  }

  /**
   * carousel block — navigable slide show of block groups.
   * { type: "carousel", slides: [{ blocks: [...] }] }
   */
  _renderCarousel(block) {
    if (!Array.isArray(block.slides) || block.slides.length === 0) return null;

    const carousel = document.createElement('div');
    carousel.className = 'block-carousel';
    carousel.dataset.carousel = '';

    const slides = [];

    // Render each slide
    for (let i = 0; i < block.slides.length; i++) {
      const slideData = block.slides[i];
      const slideEl = document.createElement('div');
      slideEl.className = 'block-carousel__slide';
      slideEl.dataset.slide = '';

      // Recursively render nested blocks inside the slide
      if (Array.isArray(slideData?.blocks)) {
        const innerFragment = this.render(slideData.blocks);
        slideEl.appendChild(innerFragment);
      }

      // Only the first slide is visible initially
      if (i !== 0) slideEl.style.display = 'none';

      carousel.appendChild(slideEl);
      slides.push(slideEl);
    }

    // Navigation controls (only needed when more than one slide)
    if (slides.length > 1) {
      // Prev / Next buttons
      const prevBtn = document.createElement('button');
      prevBtn.className = 'block-carousel__prev';
      prevBtn.dataset.prev = '';
      prevBtn.setAttribute('aria-label', 'Previous slide');
      prevBtn.textContent = '‹';

      const nextBtn = document.createElement('button');
      nextBtn.className = 'block-carousel__next';
      nextBtn.dataset.next = '';
      nextBtn.setAttribute('aria-label', 'Next slide');
      nextBtn.textContent = '›';

      // Dot indicators
      const dotsContainer = document.createElement('div');
      dotsContainer.className = 'block-carousel__dots';

      const dots = [];
      for (let i = 0; i < slides.length; i++) {
        const dot = document.createElement('button');
        dot.className = 'block-carousel__dot' + (i === 0 ? ' block-carousel__dot--active' : '');
        dot.dataset.dot = i;
        dot.setAttribute('aria-label', `Slide ${i + 1}`);
        dotsContainer.appendChild(dot);
        dots.push(dot);
      }

      let current = 0;

      const goTo = (index) => {
        slides[current].style.display = 'none';
        dots[current].classList.remove('block-carousel__dot--active');
        current = (index + slides.length) % slides.length;
        slides[current].style.display = '';
        dots[current].classList.add('block-carousel__dot--active');
      };

      prevBtn.addEventListener('click', () => goTo(current - 1));
      nextBtn.addEventListener('click', () => goTo(current + 1));

      dotsContainer.addEventListener('click', (e) => {
        const dot = e.target.closest('[data-dot]');
        if (!dot) return;
        const idx = parseInt(dot.dataset.dot, 10);
        if (!isNaN(idx)) goTo(idx);
      });

      carousel.appendChild(prevBtn);
      carousel.appendChild(nextBtn);
      carousel.appendChild(dotsContainer);
    }

    return carousel;
  }

  /**
   * Unknown block type — render fallback text if present, otherwise skip.
   * @param {Object} block
   * @returns {Element|null}
   */
  _renderUnknown(block) {
    if (typeof block.fallback === 'string' && block.fallback !== '') {
      const p = document.createElement('p');
      p.className = 'block-text';
      p.textContent = block.fallback;
      return p;
    }
    return null;
  }

  // ---------------------------------------------------------------------------
  // Inline markdown parser
  // ---------------------------------------------------------------------------

  /**
   * Parse inline markdown into safe HTML.
   *
   * Steps:
   *   1. Escape HTML entities in the raw text
   *   2. Extract code spans (protect from further transforms)
   *   3. Apply bold, italic, strikethrough, link transforms
   *   4. Restore code spans
   *   5. Convert newlines to <br>
   *
   * @param {string} text — raw text (may contain inline markdown)
   * @returns {string} — HTML string safe for innerHTML
   */
  _parseInlineMarkdown(text) {
    if (typeof text !== 'string' || text === '') return '';

    // Step 1: Escape all HTML entities first so raw < > & etc. are neutralised
    // before we inject our own HTML tags.
    let html = escHtml(text);
    // Note: escHtml() from utils.js escapes & < > " but not '.
    // The apostrophe (') is safe in text content and inside attribute values
    // we control, so we do not need to double-escape it here.

    // Step 2: Temporarily replace code spans so their contents are not
    // processed by subsequent regex transforms.
    // We use a placeholder map keyed by index.
    const codePlaceholders = [];
    html = html.replace(/`([^`]+)`/g, (_, inner) => {
      const idx = codePlaceholders.length;
      // inner is already HTML-escaped (from step 1); wrap in <code>
      codePlaceholders.push(`<code>${inner}</code>`);
      return `\x00CODE${idx}\x00`;
    });

    // Step 3a: Bold — **text** or __text__
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/__(.+?)__/g, '<strong>$1</strong>');

    // Step 3b: Italic — *text* or _text_
    // For * — match single asterisk not preceded/followed by another asterisk
    html = html.replace(/\*(?!\*)(.+?)(?<!\*)\*/g, '<em>$1</em>');
    // For _ — only match when not surrounded by word characters on both sides
    // (avoids matching snake_case identifiers)
    html = html.replace(/(?<!\w)_(?!_)(.+?)(?<!_)_(?!\w)/g, '<em>$1</em>');

    // Step 3c: Strikethrough — ~~text~~
    html = html.replace(/~~(.+?)~~/g, '<del>$1</del>');

    // Step 3d: Links — [text](url) — http/https only
    // At this point, the URL in the original text has been HTML-escaped,
    // so "https://..." becomes "https://..." (unchanged for these chars),
    // but we must reconstruct the original URL from the escaped form.
    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_, linkText, href) => {
      // Unescape any &amp; that escHtml may have introduced in the URL
      const rawHref = href.replace(/&amp;/g, '&').trim();
      if (!isSafeUrl(rawHref)) {
        // Unsafe URL — render just the link text
        return linkText;
      }
      // Re-escape the href for use in the attribute value
      const safeHref = escHtml(rawHref);
      return `<a href="${safeHref}" target="_blank" rel="noopener noreferrer">${linkText}</a>`;
    });

    // Step 4: Restore code span placeholders
    html = html.replace(/\x00CODE(\d+)\x00/g, (_, idx) => codePlaceholders[parseInt(idx, 10)]);

    // Step 5: Newlines → <br>
    html = html.replace(/\n/g, '<br>');

    return html;
  }
}
