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
      case 'columns':   return this._renderColumns(block);
      case 'section':   return this._renderSection(block);
      case 'tabs':      return this._renderTabs(block);
      case 'container': return this._renderContainer(block);
      case 'input':     return this._renderInput(block);
      case 'select':    return this._renderSelect(block);
      case 'toggle':    return this._renderToggle(block);
      case 'form':      return this._renderForm(block);
      case 'badge':     return this._renderBadge(block);
      case 'alert':     return this._renderAlert(block);
      case 'loading':   return this._renderLoading(block);
      case 'thought':   return this._renderThought(block);
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
   * actions block — action buttons.
   * Chat actions: { label, skill, payload } — one-time use, dispatches chalie:action
   * Interface actions: { label, execute, collect?, target?, openUrl?, style? } — daemon capability calls
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
      if (btn.style === 'secondary') button.classList.add('block-action-btn--secondary');
      if (btn.style === 'danger') button.classList.add('block-action-btn--danger');
      button.textContent = label;

      // Interface daemon action: execute a capability via gateway
      if (btn.execute) {
        button.dataset.execute = btn.execute;
        if (btn.collect) button.dataset.collect = btn.collect;
        if (btn.target) button.dataset.target = btn.target;
        if (btn.openUrl) button.dataset.openUrl = 'true';
        if (btn.payload) button.dataset.payload = JSON.stringify(btn.payload);
        // Click handling is wired by the overlay host (apps_panel / app.js),
        // not here — the block renderer has no gateway context.
      } else {
        // Chat action: one-time use, dispatches chalie:action
        button.addEventListener('click', () => {
          for (const b of container.querySelectorAll('.block-action-btn')) {
            b.disabled = true;
          }
          button.classList.add('block-action-btn--selected');

          document.dispatchEvent(new CustomEvent('chalie:action', {
            detail: { payload: btn.payload ?? null },
          }));
        });
      }

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

  // ---------------------------------------------------------------------------
  // Interface block renderers (P5 — layout, form, feedback)
  // ---------------------------------------------------------------------------

  /**
   * columns block — CSS Grid multi-column layout.
   * { type: "columns", columns: [{ width: "1fr", blocks: [...] }, ...] }
   */
  _renderColumns(block) {
    if (!Array.isArray(block.columns) || block.columns.length === 0) return null;

    const grid = document.createElement('div');
    grid.className = 'block-columns';
    grid.style.gridTemplateColumns = block.columns
      .map(c => c.width || '1fr')
      .join(' ');

    for (const col of block.columns) {
      const cell = document.createElement('div');
      cell.className = 'block-columns__col';
      if (Array.isArray(col.blocks)) {
        cell.appendChild(this.render(col.blocks));
      }
      grid.appendChild(cell);
    }

    return grid;
  }

  /**
   * section block — grouped content with optional title and collapsibility.
   * { type: "section", title?: "...", collapsible?: false, blocks: [...] }
   */
  _renderSection(block) {
    if (!Array.isArray(block.blocks)) return null;

    if (block.collapsible) {
      const details = document.createElement('details');
      details.className = 'block-section';
      details.open = true;
      if (block.title) {
        const summary = document.createElement('summary');
        summary.className = 'block-section__title';
        summary.textContent = block.title;
        details.appendChild(summary);
      }
      const body = document.createElement('div');
      body.className = 'block-section__body';
      body.appendChild(this.render(block.blocks));
      details.appendChild(body);
      return details;
    }

    const div = document.createElement('div');
    div.className = 'block-section';
    if (block.title) {
      const h = document.createElement('div');
      h.className = 'block-section__title';
      h.textContent = block.title;
      div.appendChild(h);
    }
    const body = document.createElement('div');
    body.className = 'block-section__body';
    body.appendChild(this.render(block.blocks));
    div.appendChild(body);
    return div;
  }

  /**
   * tabs block — tabbed panels.
   * { type: "tabs", tabs: [{ label: "Tab 1", blocks: [...] }, ...] }
   */
  _renderTabs(block) {
    if (!Array.isArray(block.tabs) || block.tabs.length === 0) return null;

    const wrapper = document.createElement('div');
    wrapper.className = 'block-tabs';

    const bar = document.createElement('div');
    bar.className = 'block-tabs__bar';

    const panels = [];
    const buttons = [];

    for (let i = 0; i < block.tabs.length; i++) {
      const tab = block.tabs[i];
      const btn = document.createElement('button');
      btn.className = 'block-tabs__btn' + (i === 0 ? ' block-tabs__btn--active' : '');
      btn.textContent = tab.label || `Tab ${i + 1}`;
      btn.dataset.tabIndex = i;
      bar.appendChild(btn);
      buttons.push(btn);

      const panel = document.createElement('div');
      panel.className = 'block-tabs__panel';
      if (i !== 0) panel.style.display = 'none';
      if (Array.isArray(tab.blocks)) {
        panel.appendChild(this.render(tab.blocks));
      }
      panels.push(panel);
    }

    bar.addEventListener('click', (e) => {
      const btn = e.target.closest('[data-tab-index]');
      if (!btn) return;
      const idx = parseInt(btn.dataset.tabIndex, 10);
      if (isNaN(idx)) return;
      for (let i = 0; i < panels.length; i++) {
        panels[i].style.display = i === idx ? '' : 'none';
        buttons[i].classList.toggle('block-tabs__btn--active', i === idx);
      }
    });

    wrapper.appendChild(bar);
    for (const p of panels) wrapper.appendChild(p);
    return wrapper;
  }

  /**
   * container block — named container for dynamic content replacement.
   * { type: "container", id: "...", blocks: [...], poll?: { capability, interval, params } }
   */
  _renderContainer(block) {
    const div = document.createElement('div');
    div.className = 'block-container';
    if (block.id) div.dataset.containerId = block.id;

    if (block.poll && typeof block.poll === 'object') {
      if (block.poll.capability) div.dataset.pollCapability = block.poll.capability;
      if (block.poll.interval) div.dataset.pollInterval = block.poll.interval;
      if (block.poll.params) div.dataset.pollParams = JSON.stringify(block.poll.params);
    }

    if (Array.isArray(block.blocks)) {
      div.appendChild(this.render(block.blocks));
    }
    return div;
  }

  /**
   * input block — text input field.
   * { type: "input", name: "...", placeholder?: "...", value?: "...", inputType?: "text" }
   */
  _renderInput(block) {
    const name = block.name;
    if (typeof name !== 'string' || !name) return null;

    const input = document.createElement('input');
    input.className = 'block-input';
    input.type = block.inputType || 'text';
    input.dataset.name = name;
    input.dataset.formField = '';
    if (block.placeholder) input.placeholder = block.placeholder;
    if (block.value != null) input.value = block.value;
    return input;
  }

  /**
   * select block — dropdown select.
   * { type: "select", name: "...", options: [{ label, value }], value?: "..." }
   */
  _renderSelect(block) {
    const name = block.name;
    if (typeof name !== 'string' || !name) return null;
    if (!Array.isArray(block.options)) return null;

    const select = document.createElement('select');
    select.className = 'block-select';
    select.dataset.name = name;
    select.dataset.formField = '';

    for (const opt of block.options) {
      if (!opt || typeof opt !== 'object') continue;
      const option = document.createElement('option');
      option.value = opt.value ?? '';
      option.textContent = opt.label || opt.value || '';
      if (block.value != null && String(opt.value) === String(block.value)) {
        option.selected = true;
      }
      select.appendChild(option);
    }

    return select;
  }

  /**
   * toggle block — on/off switch.
   * { type: "toggle", name: "...", label: "...", value?: false }
   */
  _renderToggle(block) {
    const name = block.name;
    if (typeof name !== 'string' || !name) return null;

    const wrapper = document.createElement('div');
    wrapper.className = 'block-toggle';

    const btn = document.createElement('button');
    btn.className = 'block-toggle__switch' + (block.value ? ' on' : '');
    btn.dataset.name = name;
    btn.dataset.formField = '';
    btn.dataset.value = block.value ? 'true' : 'false';
    btn.addEventListener('click', () => {
      const isOn = btn.classList.toggle('on');
      btn.dataset.value = isOn ? 'true' : 'false';
    });

    const label = document.createElement('span');
    label.className = 'block-toggle__label';
    label.textContent = block.label || name;

    wrapper.appendChild(btn);
    wrapper.appendChild(label);
    return wrapper;
  }

  /**
   * form block — groups input/select/toggle fields for collection.
   * { type: "form", id: "...", blocks: [...] }
   */
  _renderForm(block) {
    if (!block.id || !Array.isArray(block.blocks)) return null;

    const div = document.createElement('div');
    div.className = 'block-form';
    div.dataset.formId = block.id;
    div.appendChild(this.render(block.blocks));
    return div;
  }

  /**
   * badge block — inline status label.
   * { type: "badge", text: "...", variant?: "info"|"success"|"warning"|"error" }
   */
  _renderBadge(block) {
    if (typeof block.text !== 'string' || !block.text) return null;

    const span = document.createElement('span');
    span.className = 'block-badge';
    if (block.variant) span.classList.add(`block-badge--${block.variant}`);
    span.textContent = block.text;
    return span;
  }

  /**
   * alert block — message banner with variant.
   * { type: "alert", message: "...", variant?: "info"|"success"|"warning"|"error" }
   */
  _renderAlert(block) {
    if (typeof block.message !== 'string' || !block.message) return null;

    const div = document.createElement('div');
    div.className = 'block-alert';
    if (block.variant) div.classList.add(`block-alert--${block.variant}`);
    div.textContent = block.message;
    return div;
  }

  /**
   * loading block — spinner with optional label.
   * { type: "loading", label?: "..." }
   */
  _renderLoading(block) {
    const div = document.createElement('div');
    div.className = 'block-loading';

    const spinner = document.createElement('div');
    spinner.className = 'block-loading__spinner';
    div.appendChild(spinner);

    if (block.label) {
      const label = document.createElement('span');
      label.className = 'block-loading__label';
      label.textContent = block.label;
      div.appendChild(label);
    }

    return div;
  }

  // ---------------------------------------------------------------------------
  // Proactive thought card
  // ---------------------------------------------------------------------------

  /**
   * thought block — proactive card surfacing a Chalie-initiated idea or
   * cross-topic connection.
   *
   * Expected block shape:
   * {
   *   type:    "thought",
   *   id:      string,        // required — used in CustomEvent detail
   *   content: string,        // required — body text of the thought
   *   context: string|null    // optional — additional context line
   * }
   *
   * Emitted events (on document):
   *   chalie:thought-action  { action: "expand",  blockId: block.id }
   *   chalie:thought-action  { action: "dismiss", blockId: block.id }
   *
   * @param {Object} block
   * @returns {Element|null}
   */
  _renderThought(block) {
    const content = block.content;
    if (typeof content !== 'string' || content === '') return null;

    // ---- outer card --------------------------------------------------------
    const card = document.createElement('div');
    card.className = 'block-thought';
    if (block.id) card.dataset.blockId = block.id;

    // ---- header ------------------------------------------------------------
    const header = document.createElement('div');
    header.className = 'block-thought__header';
    header.textContent = 'Chalie had a thought';
    card.appendChild(header);

    // ---- body --------------------------------------------------------------
    const body = document.createElement('p');
    body.className = 'block-thought__body';
    body.textContent = content;
    card.appendChild(body);

    // ---- optional context line ---------------------------------------------
    if (block.context && typeof block.context === 'string') {
      const ctx = document.createElement('p');
      ctx.className = 'block-thought__context';
      ctx.textContent = block.context;
      card.appendChild(ctx);
    }

    // ---- action row --------------------------------------------------------
    const actions = document.createElement('div');
    actions.className = 'block-thought__actions';

    // Helper — dispatch chalie:thought-action
    const dispatch = (action) => {
      document.dispatchEvent(new CustomEvent('chalie:thought-action', {
        detail: { action, blockId: block.id ?? null },
      }));
    };

    // "Tell me more" — expand
    const expandBtn = document.createElement('button');
    expandBtn.className = 'block-thought__btn block-thought__btn--expand';
    expandBtn.textContent = 'Tell me more';
    expandBtn.addEventListener('click', () => dispatch('expand'));

    // "Not now" — dismiss (fade out the card)
    const dismissBtn = document.createElement('button');
    dismissBtn.className = 'block-thought__btn block-thought__btn--dismiss';
    dismissBtn.textContent = 'Not now';
    dismissBtn.addEventListener('click', () => {
      card.classList.add('block-thought--fading');
      dispatch('dismiss');
    });

    // "×" close — remove from DOM immediately
    const closeBtn = document.createElement('button');
    closeBtn.className = 'block-thought__btn block-thought__btn--close';
    closeBtn.setAttribute('aria-label', 'Close thought card');
    closeBtn.textContent = '×';
    closeBtn.addEventListener('click', () => card.remove());

    actions.appendChild(expandBtn);
    actions.appendChild(dismissBtn);
    actions.appendChild(closeBtn);
    card.appendChild(actions);

    return card;
  }

  // ---------------------------------------------------------------------------
  // Fallback
  // ---------------------------------------------------------------------------

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
