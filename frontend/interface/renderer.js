/**
 * Conversation spine DOM renderer.
 */
import { renderMarkupTo } from './markup_renderer.js';
import { extractPlaintext } from './markup_extract.js';
import { renderCard } from './rich_media/registry.js';

const REMEMBER_ICON = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none"
  stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <path d="M12 2l2.09 6.26L20 10l-5.91 1.74L12 18l-2.09-6.26L4 10l5.91-1.74L12 2z"></path>
</svg>`;

const SPEAK_ICON = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none"
  stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
  <path d="M15.54 8.46a5 5 0 0 1 0 7.07"></path>
  <path d="M19.07 4.93a10 10 0 0 1 0 14.14"></path>
</svg>`;

// Sender glyph — fused at the cap of the accent stripe.
// Constellation (4 dots + 3 hairlines) for Chalie, single spark for the user.
const CHALIE_GLYPH = `<svg class="sender-glyph" viewBox="0 0 18 18" fill="none" aria-hidden="true">
  <line x1="4" y1="14" x2="9" y2="5" stroke="currentColor" stroke-width="0.6" opacity="0.45"></line>
  <line x1="9" y1="5" x2="14" y2="11" stroke="currentColor" stroke-width="0.6" opacity="0.45"></line>
  <line x1="4" y1="14" x2="14" y2="11" stroke="currentColor" stroke-width="0.6" opacity="0.45"></line>
  <circle class="sender-glyph__dot" cx="4" cy="14" r="1.4" fill="currentColor"></circle>
  <circle class="sender-glyph__dot" cx="9" cy="5" r="1.6" fill="currentColor"></circle>
  <circle class="sender-glyph__dot" cx="14" cy="11" r="1.3" fill="currentColor"></circle>
  <circle class="sender-glyph__dot" cx="11" cy="2" r="0.9" fill="currentColor" opacity="0.65"></circle>
</svg>`;

const USER_GLYPH = `<svg class="sender-glyph" viewBox="0 0 18 18" fill="none" role="img" aria-label="You said">
  <title>You said</title>
  <path d="M9 2 L11 7 L16 9 L11 11 L9 16 L7 11 L2 9 L7 7 Z" fill="currentColor" opacity="0.85"></path>
</svg>`;

function _attachGlyph(el, role) {
  const wrapper = document.createElement('template');
  wrapper.innerHTML = role === 'chalie' ? CHALIE_GLYPH : USER_GLYPH;
  el.appendChild(wrapper.content.firstElementChild);
}

export class Renderer {
  /**
   * @param {HTMLElement} spine — the .conversation-spine element
   */
  constructor(spine) {
    this._spine = spine;
    this._userScrolledUp = false;
    this._activeForm = null;
    this._initScrollTracking();
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  /**
   * Append a user speech form.
   * @param {string} text
   * @param {string|null} [ts]
   * @param {{inWorkingMemory?: boolean, attachments?: Array}} [options]
   */
  appendUserForm(text, ts = null, { inWorkingMemory = true, attachments = [] } = {}) {
    const el = this._createEl('div', 'speech-form speech-form--user');
    if (!inWorkingMemory) el.classList.add('message--faded');
    _attachGlyph(el, 'user');

    this._appendUserAttachments(el, attachments);

    // Suppress the '[File attached]' placeholder when there is nothing typed but
    // the attachments are already shown above — the media is the message.
    if (!(text === '[File attached]' && attachments.length)) {
      const textEl = this._createEl('div', 'speech-form__text');
      textEl.textContent = text;
      el.appendChild(textEl);
    }

    this._spine.appendChild(el);
    this._scrollToBottom();
    return el;
  }

  /**
   * Render attachment previews (image thumbnails / doc chips) inside a user
   * speech-form, above the text. No-op when there are no attachments.
   *
   * @param {HTMLElement} el — the .speech-form--user element
   * @param {Array<{filename: string, objectUrl: string|null, isImage: boolean}>} attachments
   */
  _appendUserAttachments(el, attachments = []) {
    if (!attachments.length) return;

    const wrap = this._createEl('div', 'speech-form__attachments');
    for (const att of attachments) {
      if (att.isImage && att.objectUrl) {
        const img = document.createElement('img');
        img.className = 'speech-form__attachment-img';
        img.src = att.objectUrl;
        img.alt = att.filename || 'attached image';
        img.loading = 'lazy';
        wrap.appendChild(img);
      } else {
        const chip = this._createEl('div', 'speech-form__attachment-doc');
        const icon = this._createEl('span', 'speech-form__attachment-doc-icon');
        icon.setAttribute('aria-hidden', 'true');
        icon.textContent = '📄';
        chip.appendChild(icon);
        const name = this._createEl('span', 'speech-form__attachment-doc-name');
        name.textContent = att.filename || 'attachment';
        chip.appendChild(name);
        wrap.appendChild(chip);
      }
    }
    el.appendChild(wrap);
  }

  /**
   * Prepend a user speech form (for scroll-up pagination).
   * @param {string} text
   * @param {string|null} [ts]
   * @param {{inWorkingMemory?: boolean, attachments?: Array}} [options]
   */
  prependUserForm(text, ts = null, { inWorkingMemory = true, attachments = [] } = {}) {
    const el = this._createEl('div', 'speech-form speech-form--user');
    if (!inWorkingMemory) el.classList.add('message--faded');
    _attachGlyph(el, 'user');

    this._appendUserAttachments(el, attachments);

    // Mirror appendUserForm: the media is the message when there's no real text.
    if (!(text === '[File attached]' && attachments.length)) {
      const textEl = this._createEl('div', 'speech-form__text');
      textEl.textContent = text;
      el.appendChild(textEl);
    }

    this._spine.prepend(el);
    return el;
  }

  /**
   * Append a Chalie speech form.
   *
   * When ``meta.segments`` is present (rich-media turn), each segment is
   * rendered independently — text segments as ordinary bubbles, rich segments
   * via the card registry.  When ``meta.segments`` is absent, falls back to
   * the original single-bubble path using ``content``.
   *
   * @param {string} content — XML markup string (kept for backwards compat)
   * @param {{segments?: Array, topic?: string, duration_ms?: number}} [meta]
   * @param {{inWorkingMemory?: boolean}} [options]
   */
  appendChalieForm(content, meta = {}, { inWorkingMemory = true } = {}) {
    const segments = meta.segments || [{ type: 'text', content: content || '' }];
    let lastEl = null;

    for (const seg of segments) {
      if (seg.type === 'text') {
        const el = this._appendTextBubble(seg.content, meta, inWorkingMemory);
        lastEl = el;
      } else if (seg.type === 'rich') {
        const el = this._appendRichCard(seg.tag, seg.payload, seg.synthesis, meta, inWorkingMemory);
        lastEl = el;
      }
    }

    if (lastEl) {
      this._setActiveForm(lastEl);
      this._scrollToBottom();
    }
    return lastEl;
  }

  /**
   * Prepend a Chalie speech form (for scroll-up pagination).
   *
   * Mirrors appendChalieForm but inserts before existing nodes (oldest
   * messages prepended on scroll-up).  Segments rendered in reverse order
   * so the first segment ends up topmost after prepend.
   *
   * @param {string} content — XML markup string (kept for backwards compat)
   * @param {{segments?: Array, topic?: string, duration_ms?: number}} [meta]
   * @param {{inWorkingMemory?: boolean}} [options]
   */
  prependChalieForm(content, meta = {}, { inWorkingMemory = true } = {}) {
    const segments = meta.segments || [{ type: 'text', content: content || '' }];
    let firstEl = null;

    // Prepend in reverse order so segments appear in correct visual sequence
    for (let i = segments.length - 1; i >= 0; i--) {
      const seg = segments[i];
      if (seg.type === 'text') {
        const el = this._prependTextBubble(seg.content, meta, inWorkingMemory);
        if (i === 0) firstEl = el;
      } else if (seg.type === 'rich') {
        const el = this._prependRichCard(seg.tag, seg.payload, seg.synthesis, meta, inWorkingMemory);
        if (i === 0) firstEl = el;
      }
    }

    return firstEl;
  }

  /**
   * Create an ACT-cycle host — chrome-less placeholder for the live tool loop.
   *
   * Structure:
   *   .act-cycle
   *     .act-row          (logo + narrative + stop button)
   *       .act-logo       (blinking violet disc, always present)
   *       .act-narrative  (italic text, mutated in-place by setActNarrative)
   *       .act-stop-btn   (cancel button; wired in Chat._startTurn)
   *     .act-tools        (cumulative tool list, populated by appendToolPill)
   *
   * No background, no border, no padding — the logo IS the placeholder.
   * On final response, replaceActWithResponse swaps the whole node for a
   * normal Chalie speech-form bubble, removing the stop button automatically.
   */
  createActCycle() {
    const el = this._createEl('div', 'act-cycle');

    const row = this._createEl('div', 'act-row');
    const logo = this._createEl('span', 'act-logo');
    const narrative = this._createEl('span', 'act-narrative');
    const stopBtn = this._createEl('button', 'act-stop-btn');
    stopBtn.setAttribute('aria-label', 'Stop and undo');
    stopBtn.title = 'Stop & undo';
    stopBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M4.5 2L1 5.5L4.5 9V6.5H10a3.5 3.5 0 0 1 0 7H7v2h3a5.5 5.5 0 0 0 0-11H4.5V2Z"/></svg>`;
    row.appendChild(logo);
    row.appendChild(narrative);
    row.appendChild(stopBtn);
    el.appendChild(row);

    const tools = this._createEl('div', 'act-tools');
    el.appendChild(tools);

    this._spine.appendChild(el);
    this._scrollToBottom();
    return el;
  }

  /**
   * Set the active narration text inside an ACT cycle.
   * Replaces the previous narrative — the spec is one-line-at-a-time, not
   * a stack of iterations. The logo stays put; only the text mutates.
   *
   * @param {HTMLElement} actEl — the .act-cycle element
   * @param {string} text — narration line from the LLM
   * @param {number} [step] — iteration number (stored as data attribute)
   */
  setActNarrative(actEl, text, step) {
    if (!actEl) return;
    const slot = actEl.querySelector(':scope > .act-row > .act-narrative');
    if (!slot) return;
    // Trust the backend chokepoint (services.markup.sanitize → nh3) — same
    // path chat bubbles use. textContent would render any LLM-emitted tags
    // (e.g. <p>) as literal characters in the inline status line.
    renderMarkupTo(slot, text || '');
    if (step != null) slot.dataset.step = String(step);
    this._scrollToBottom();
  }

  /**
   * Append a tool row to an ACT cycle's cumulative tool list.
   * Tool rows span the entire ACT loop — they are NOT nested under narrations.
   *
   * @param {HTMLElement} actEl — the .act-cycle element
   * @param {string} callId — server-assigned id for resolveToolPill lookup
   * @param {string} name — tool name to display
   * @param {string|undefined} summary — optional ~3-10 word description of the call
   * @returns {HTMLElement|null} the row element (null if actEl falsy)
   */
  appendToolPill(actEl, callId, name, summary) {
    if (!actEl || !callId) return null;
    const host = actEl.querySelector(':scope > .act-tools');
    if (!host) return null;

    const row = this._createEl('div', 'act-tool act-tool--running');
    row.dataset.callId = callId;
    row.dataset.startedAt = String(Date.now());

    const nameEl = this._createEl('span', 'act-tool__name');
    nameEl.textContent = name || 'tool';

    const statusEl = this._createEl('span', 'act-tool__status');
    const spinner = this._createEl('span', 'act-spinner');
    statusEl.appendChild(spinner);

    if (summary) {
      const labelEl = this._createEl('span', 'act-tool__label');
      labelEl.appendChild(nameEl);
      const summaryEl = this._createEl('span', 'act-tool__summary');
      summaryEl.textContent = `— ${summary}`;
      labelEl.appendChild(summaryEl);
      row.appendChild(labelEl);
    } else {
      row.appendChild(nameEl);
    }
    row.appendChild(statusEl);
    host.appendChild(row);
    this._scrollToBottom();
    return row;
  }

  /**
   * Resolve a tool row — spinner → duration (ok) or literal "error" (!ok).
   * Enforces a 150ms minimum visible duration so sub-100ms tools still flash
   * the spinner before settling.
   *
   * @param {string} callId
   * @param {number} ms — server-reported elapsed ms (>= 0)
   * @param {boolean} ok
   */
  resolveToolPill(callId, ms, ok) {
    if (!callId) return;
    const row = this._spine.querySelector(
      `.act-tool[data-call-id="${CSS.escape(callId)}"]`
    );
    if (!row) return;

    const startedAt = Number.parseInt(row.dataset.startedAt || '0', 10);
    const elapsed = startedAt ? Date.now() - startedAt : 200;
    const wait = Math.max(0, 150 - elapsed);
    // The backend act_tool_end event carries no `ms`; fall back to the
    // client-measured elapsed so the pill still shows a real duration.
    const effectiveMs = Number(ms) > 0
      ? Number(ms)
      : (startedAt ? Date.now() - startedAt : 0);

    setTimeout(() => {
      row.classList.remove('act-tool--running');
      row.classList.add(ok ? 'act-tool--done' : 'act-tool--error');
      const statusEl = row.querySelector('.act-tool__status');
      if (!statusEl) return;
      statusEl.innerHTML = '';
      if (ok) {
        const seconds = (Math.max(0, effectiveMs) / 1000).toFixed(1);
        statusEl.textContent = `${seconds}s`;
      } else {
        statusEl.textContent = 'error';
      }
    }, wait);
  }

  /**
   * Replace an ACT cycle with a final Chalie response bubble.
   * The act-cycle node is removed; a fresh .speech-form--chalie is appended
   * in its place. This is the spec-mandated transition: ACT UI vanishes
   * entirely on completion, replaced by a normal chat bubble.
   *
   * @param {HTMLElement} actEl — the .act-cycle element
   * @param {string} content — XML markup string
   * @param {{topic?: string, duration_ms?: number}} [meta]
   */
  replaceActWithResponse(actEl, content, meta = {}) {
    if (actEl?.isConnected) actEl.remove();
    return this.appendChalieForm(content, meta);
  }

  /**
   * Replace an ACT cycle with an error speech-form.
   * @param {HTMLElement} actEl — the .act-cycle element
   * @param {string} message
   */
  replaceActWithError(actEl, message) {
    if (actEl?.isConnected) actEl.remove();
    const el = this._createEl('div', 'speech-form speech-form--chalie speech-form--error');
    el.appendChild(this._buildChalieHeader({}));
    const textEl = this._createEl('div', 'speech-form__text');
    textEl.textContent = message;
    el.appendChild(textEl);
    this._spine.appendChild(el);
    this._scrollToBottom();
    return el;
  }

  /** Remove all children from the spine. */
  clear() {
    this._spine.innerHTML = '';
  }

  // ---------------------------------------------------------------------------
  // Private helpers
  // ---------------------------------------------------------------------------

  /** Append a single text bubble to the spine. Returns the element. */
  _appendTextBubble(content, meta, inWorkingMemory) {
    const el = this._createEl('div', 'speech-form speech-form--chalie');
    if (!inWorkingMemory) el.classList.add('message--faded');
    el.appendChild(this._buildChalieHeader(meta));
    const textEl = this._createEl('div', 'speech-form__text');
    renderMarkupTo(textEl, content || '');
    el.appendChild(textEl);
    const metaRow = this._buildMetaRow(this._extractPlainText(content), meta);
    el.appendChild(metaRow);
    this._spine.appendChild(el);
    return el;
  }

  /** Prepend a single text bubble. Returns the element. */
  _prependTextBubble(content, meta, inWorkingMemory) {
    const el = this._createEl('div', 'speech-form speech-form--chalie');
    if (!inWorkingMemory) el.classList.add('message--faded');
    el.appendChild(this._buildChalieHeader(meta));
    const textEl = this._createEl('div', 'speech-form__text');
    renderMarkupTo(textEl, content || '');
    el.appendChild(textEl);
    const metaRow = this._buildMetaRow(this._extractPlainText(content), meta);
    el.appendChild(metaRow);
    this._spine.prepend(el);
    return el;
  }

  /** Append a rich-media card to the spine. Returns the card wrapper element. */
  _appendRichCard(tag, payload, synthesis, meta, inWorkingMemory) {
    const wrapper = this._createEl('div', 'rich-card-wrapper');
    if (!inWorkingMemory) wrapper.classList.add('message--faded');

    const fallback = (synText) => {
      const fallbackEl = this._createEl('div', 'speech-form speech-form--chalie');
      const textEl = this._createEl('div', 'speech-form__text');
      renderMarkupTo(textEl, synText || '');
      fallbackEl.appendChild(textEl);
      this._spine.appendChild(fallbackEl);
    };

    renderCard(tag, payload, synthesis, wrapper, fallback);
    this._spine.appendChild(wrapper);
    return wrapper;
  }

  /** Prepend a rich-media card. Returns the card wrapper element. */
  _prependRichCard(tag, payload, synthesis, meta, inWorkingMemory) {
    const wrapper = this._createEl('div', 'rich-card-wrapper');
    if (!inWorkingMemory) wrapper.classList.add('message--faded');

    const fallback = (synText) => {
      const fallbackEl = this._createEl('div', 'speech-form speech-form--chalie');
      const textEl = this._createEl('div', 'speech-form__text');
      renderMarkupTo(textEl, synText || '');
      fallbackEl.appendChild(textEl);
      this._spine.prepend(fallbackEl);
    };

    renderCard(tag, payload, synthesis, wrapper, fallback);
    this._spine.prepend(wrapper);
    return wrapper;
  }

  _buildChalieHeader(meta) {
    const header = this._createEl('div', 'speech-form__header');
    const wrapper = document.createElement('template');
    wrapper.innerHTML = CHALIE_GLYPH;
    header.appendChild(wrapper.content.firstElementChild);
    const timestampEl = this._createEl('span', 'speech-form__timestamp');
    timestampEl.textContent = meta.ts || '';
    header.appendChild(timestampEl);
    return header;
  }

  _buildMetaRow(text, meta) {
    const MODE_LABELS = { ACT: 'acting', CLARIFY: 'clarifying', ACKNOWLEDGE: 'noting' };

    const metaRow = this._createEl('div', 'speech-form__meta');

    // Mode badge — only for non-default modes (skip RESPOND)
    if (meta.mode && MODE_LABELS[meta.mode]) {
      const badge = document.createElement('span');
      badge.className = 'meta-mode-badge';
      badge.textContent = MODE_LABELS[meta.mode];
      metaRow.appendChild(badge);
    }

    // Remember (pin) button — always shown on Chalie messages
    const rememberBtn = this._createEl('button', 'speech-form__remember-btn');
    rememberBtn.setAttribute('aria-label', 'Remember this');
    rememberBtn.innerHTML = REMEMBER_ICON;
    rememberBtn.addEventListener('click', () => {
      if (rememberBtn.disabled) return;
      // 150ms micro-delay before activating glow (feels organic)
      rememberBtn.disabled = true;
      setTimeout(() => {
        document.dispatchEvent(new CustomEvent('chalie:pin-moment', {
          detail: { text, meta }
        }));
        rememberBtn.classList.add('speech-form__remember-btn--active');
      }, 150);
    });
    metaRow.appendChild(rememberBtn);

    if (text) {
      const speakBtn = this._createEl('button', 'speech-form__speak-btn');
      speakBtn.setAttribute('aria-label', 'Listen to this message');
      speakBtn.innerHTML = SPEAK_ICON;
      speakBtn.addEventListener('click', () => {
        // Dispatch synchronously inside the click gesture so VoicePlayer
        // can call AudioContext.resume() within the same user-activation
        // window (iOS Safari autoplay gate).
        document.dispatchEvent(new CustomEvent('chalie:speak-message', {
          detail: { text }
        }));
      });
      metaRow.appendChild(speakBtn);
    }

    return metaRow;
  }

  _extractPlainText(content) {
    return extractPlaintext(content || '');
  }

  _createEl(tag, className) {
    const el = document.createElement(tag);
    if (className) el.className = className;
    return el;
  }

  _initScrollTracking() {
    window.addEventListener('scroll', () => {
      const scrollBottom = document.documentElement.scrollHeight - window.scrollY - window.innerHeight;
      this._userScrolledUp = scrollBottom > 100;
      this._updateAmbientBloom();
    }, { passive: true });
    window.addEventListener('resize', () => this._updateAmbientBloom(), { passive: true });
  }

  /**
   * Reactive ambient bloom — pick the speech-form whose top edge is highest
   * above the viewport mid-line and tag <body> with .speaker-user / .speaker-chalie
   * accordingly. CSS in #ambientBloom fades a violet/cyan tint to match.
   * No-op when no forms are visible yet (keeps the canvas neutral on cold start).
   */
  _updateAmbientBloom() {
    const forms = this._spine?.querySelectorAll(':scope > .speech-form');
    if (!forms || forms.length === 0) {
      document.body.classList.remove('speaker-user', 'speaker-chalie');
      return;
    }
    const midline = window.innerHeight * 0.55;
    let active = null;
    for (const f of forms) {
      if (f.getBoundingClientRect().top < midline) active = f;
    }
    document.body.classList.toggle('speaker-user',   !!active && active.classList.contains('speech-form--user'));
    document.body.classList.toggle('speaker-chalie', !!active && active.classList.contains('speech-form--chalie'));
  }

  _scrollToBottom() {
    if (this._userScrolledUp) {
      this._updateAmbientBloom();
      return;
    }
    requestAnimationFrame(() => {
      window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
      this._updateAmbientBloom();
    });
  }

  /**
   * Force-scroll to the very bottom, ignoring the _userScrolledUp guard.
   *
   * Defers to two rAFs so the first scroll fires after the browser has
   * applied layout for any just-appended nodes, and re-fires once each
   * in-spine image finishes loading — images arrive async and shift
   * scrollHeight after the initial scroll, leaving the view mid-page
   * on page refresh. A single `instant` scroll before layout settles
   * was the original bug.
   */
  forceScrollToBottom() {
    this._userScrolledUp = false;
    const scroll = () => {
      window.scrollTo({ top: document.body.scrollHeight, behavior: 'instant' });
    };
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        scroll();
        const imgs = this._spine?.querySelectorAll('img') || [];
        for (const img of imgs) {
          if (img.complete) continue;
          // Late-loading image; re-scroll once it lands. Guarded by
          // `_userScrolledUp` so we don't yank the user if they've
          // since scrolled up.
          const retry = () => {
            if (!this._userScrolledUp) scroll();
          };
          img.addEventListener('load', retry, { once: true });
          img.addEventListener('error', retry, { once: true });
        }
      });
    });
  }

  _setActiveForm(el) {
    if (this._activeForm) {
      this._activeForm.classList.remove('speech-form--active');
    }
    el.classList.add('speech-form--active');
    this._activeForm = el;
  }
}
