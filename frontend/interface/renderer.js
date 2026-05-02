/**
 * Conversation spine DOM renderer.
 */
import { renderMarkupTo } from './markup_renderer.js';
import { extractPlaintext } from './markup_extract.js';

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
   * @param {{inWorkingMemory?: boolean}} [options]
   * @param {string[]} [imageIds]
   */
  appendUserForm(text, ts = null, { inWorkingMemory = true } = {}, imageIds = []) {
    const el = this._createEl('div', 'speech-form speech-form--user');
    if (!inWorkingMemory) el.classList.add('message--faded');

    // Render image thumbnails above the text when images are attached
    if (imageIds.length > 0) {
      const imagesEl = this._createEl('div', 'speech-form__images');
      for (const id of imageIds) {
        const img = document.createElement('img');
        img.src = `/chat/image/${id}/file`;
        img.className = 'speech-form__image-thumb';
        img.alt = 'Attached image';
        imagesEl.appendChild(img);
      }
      el.appendChild(imagesEl);
    }

    const textEl = this._createEl('div', 'speech-form__text');
    const isPlaceholder = text === '[Image attached]' || text === '[image attached]';
    if (isPlaceholder && imageIds.length > 0) {
      textEl.style.display = 'none';
    }
    textEl.textContent = text;
    el.appendChild(textEl);

    // Append meta row with timestamp
    const metaRow = this._createEl('div', 'speech-form__meta');
    const timestampEl = this._createEl('span', 'speech-form__timestamp');
    timestampEl.textContent = this._formatTimestamp(ts);
    metaRow.appendChild(timestampEl);
    el.appendChild(metaRow);

    this._spine.appendChild(el);
    this._scrollToBottom();
    return el;
  }

  /**
   * Prepend a user speech form (for scroll-up pagination).
   * @param {string} text
   * @param {string|null} [ts]
   * @param {{inWorkingMemory?: boolean}} [options]
   */
  prependUserForm(text, ts = null, { inWorkingMemory = true } = {}) {
    const el = this._createEl('div', 'speech-form speech-form--user');
    if (!inWorkingMemory) el.classList.add('message--faded');
    const textEl = this._createEl('div', 'speech-form__text');
    textEl.textContent = text;
    el.appendChild(textEl);

    const metaRow = this._createEl('div', 'speech-form__meta');
    const timestampEl = this._createEl('span', 'speech-form__timestamp');
    timestampEl.textContent = this._formatTimestamp(ts);
    metaRow.appendChild(timestampEl);
    el.appendChild(metaRow);

    this._spine.prepend(el);
    return el;
  }

  /**
   * Append a Chalie speech form.
   * @param {string} content — XML markup string
   * @param {{topic?: string, duration_ms?: number}} [meta]
   * @param {{inWorkingMemory?: boolean}} [options]
   */
  appendChalieForm(content, meta = {}, { inWorkingMemory = true } = {}) {
    const el = this._createEl('div', 'speech-form speech-form--chalie');
    if (!inWorkingMemory) el.classList.add('message--faded');
    const textEl = this._createEl('div', 'speech-form__text');
    renderMarkupTo(textEl, content || '');
    el.appendChild(textEl);

    const metaRow = this._buildMetaRow(this._extractPlainText(content), meta);
    el.appendChild(metaRow);

    this._spine.appendChild(el);
    this._setActiveForm(el);
    this._scrollToBottom();
    return el;
  }

  /**
   * Prepend a Chalie speech form (for scroll-up pagination).
   * @param {string} content — XML markup string
   * @param {{topic?: string, duration_ms?: number}} [meta]
   * @param {{inWorkingMemory?: boolean}} [options]
   */
  prependChalieForm(content, meta = {}, { inWorkingMemory = true } = {}) {
    const el = this._createEl('div', 'speech-form speech-form--chalie');
    if (!inWorkingMemory) el.classList.add('message--faded');
    const textEl = this._createEl('div', 'speech-form__text');
    renderMarkupTo(textEl, content || '');
    el.appendChild(textEl);

    const metaRow = this._buildMetaRow(this._extractPlainText(content), meta);
    el.appendChild(metaRow);

    this._spine.prepend(el);
    return el;
  }

  /**
   * Create an ACT-cycle host — chrome-less placeholder for the live tool loop.
   *
   * Structure:
   *   .act-cycle
   *     .act-row          (logo + narrative line)
   *       .act-logo       (blinking violet disc, always present)
   *       .act-narrative  (italic text, mutated in-place by setActNarrative)
   *     .act-tools        (cumulative tool list, populated by appendToolPill)
   *
   * No background, no border, no padding — the logo IS the placeholder.
   * On final response, replaceActWithResponse swaps the whole node for a
   * normal Chalie speech-form bubble.
   */
  createActCycle() {
    const el = this._createEl('div', 'act-cycle');

    const row = this._createEl('div', 'act-row');
    const logo = this._createEl('span', 'act-logo');
    const narrative = this._createEl('span', 'act-narrative');
    row.appendChild(logo);
    row.appendChild(narrative);
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
    slot.textContent = text || '';
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
   * @returns {HTMLElement|null} the row element (null if actEl falsy)
   */
  appendToolPill(actEl, callId, name) {
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

    row.appendChild(nameEl);
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

    setTimeout(() => {
      row.classList.remove('act-tool--running');
      row.classList.add(ok ? 'act-tool--done' : 'act-tool--error');
      const statusEl = row.querySelector('.act-tool__status');
      if (!statusEl) return;
      statusEl.innerHTML = '';
      if (ok) {
        const seconds = (Math.max(0, Number(ms) || 0) / 1000).toFixed(1);
        statusEl.textContent = `${seconds}s`;
      } else {
        statusEl.textContent = 'error';
      }
    }, wait);
  }

  /**
   * Append a steer bubble inside the in-progress ACT cycle's tool list, so the
   * user's mid-ACT redirect renders as part of the tool-calling section rather
   * than as a separate chat bubble. Falls back to the spine if no live ACT.
   *
   * @param {string} text — the user's steering message
   * @param {HTMLElement} [actEl] — the .act-cycle element for the running turn
   */
  appendSteerBubble(text, actEl) {
    const bubble = this._createEl('div', 'steer-bubble');
    bubble.textContent = text;

    const tools = actEl?.isConnected
      ? actEl.querySelector(':scope > .act-tools')
      : null;
    if (tools) {
      tools.appendChild(bubble);
    } else {
      this._spine.appendChild(bubble);
    }
    this._scrollToBottom();
    return bubble;
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

  _buildMetaRow(text, meta) {
    const MODE_LABELS = { ACT: 'acting', CLARIFY: 'clarifying', ACKNOWLEDGE: 'noting' };

    const metaRow = this._createEl('div', 'speech-form__meta');

    // Timestamp — always shown
    const timestampEl = this._createEl('span', 'speech-form__timestamp');
    timestampEl.textContent = this._formatTimestamp(meta.ts ?? null);
    metaRow.appendChild(timestampEl);

    // Mode badge — only for non-default modes (skip RESPOND)
    if (meta.mode && MODE_LABELS[meta.mode]) {
      const badge = document.createElement('span');
      badge.className = 'meta-mode-badge';
      badge.textContent = MODE_LABELS[meta.mode];
      metaRow.appendChild(badge);
    }

    // Confidence dot — color reflects routing confidence
    if (meta.confidence > 0) {
      const dot = document.createElement('span');
      dot.className = 'meta-confidence-dot';
      const c = meta.confidence;
      dot.classList.add(c >= 0.85 ? '--high' : c >= 0.65 ? '--mid' : '--low');
      const label = c >= 0.85 ? 'Highly' : c >= 0.65 ? 'Moderately' : 'Less';
      dot.title = `${label} confident (${Math.round(c * 100)}%)`;
      metaRow.appendChild(dot);
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
    });
  }

  _scrollToBottom() {
    if (this._userScrolledUp) return;
    requestAnimationFrame(() => {
      window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
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

  _formatTimestamp(ts) {
    const d = ts ? new Date(ts) : new Date();
    const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
    const day = String(d.getDate()).padStart(2, '0');
    return `${day} ${months[d.getMonth()]} ${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}`;
  }

  _setActiveForm(el) {
    if (this._activeForm) {
      this._activeForm.classList.remove('speech-form--active');
    }
    el.classList.add('speech-form--active');
    this._activeForm = el;
  }
}
