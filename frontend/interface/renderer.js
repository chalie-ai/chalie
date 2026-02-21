/**
 * Conversation spine DOM renderer.
 */

const SPEAK_ICON = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
  <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
  <path d="M19.07 4.93a10 10 0 0 1 0 14.14"></path>
  <path d="M15.54 8.46a5 5 0 0 1 0 7.07"></path>
</svg>`;

export class Renderer {
  /**
   * @param {HTMLElement} spine — the .conversation-spine element
   */
  constructor(spine) {
    this._spine = spine;
    this._userScrolledUp = false;
    this._messagesByRemovalId = new Map(); // Track messages by removed_by ID
    this._ttsEnabled = false;
    this._activeForm = null;
    this._initScrollTracking();
  }

  /** Enable or disable the TTS speaker button on Chalie messages. */
  setTtsEnabled(enabled) {
    this._ttsEnabled = enabled;
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  /** Append a user speech form. */
  appendUserForm(text, ts = null) {
    const el = this._createEl('div', 'speech-form speech-form--user');
    const textEl = this._createEl('div', 'speech-form__text');
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
   * Append a Chalie speech form.
   * @param {string} text
   * @param {{topic?: string, duration_ms?: number, removed_by?: string, removes?: string}} [meta]
   */
  appendChalieForm(text, meta = {}) {
    const el = this._createEl('div', 'speech-form speech-form--chalie');
    const textEl = this._createEl('div', 'speech-form__text');
    textEl.textContent = text;
    el.appendChild(textEl);

    const metaRow = this._buildMetaRow(text, meta);
    el.appendChild(metaRow);

    // Handle removed_by parameter — mark for pending removal
    if (meta.removed_by) {
      el.classList.add('speech-form--pending-removal');
      this._messagesByRemovalId.set(meta.removed_by, el);
    }

    // Handle removes parameter — delete messages with matching removed_by
    if (meta.removes) {
      const removedEl = this._messagesByRemovalId.get(meta.removes);
      if (removedEl) {
        removedEl.remove();
        this._messagesByRemovalId.delete(meta.removes);
      }
    }

    this._spine.appendChild(el);
    this._setActiveForm(el);
    this._scrollToBottom();
    return el;
  }

  /** Create a pending (thinking dots) form. Returns the element. */
  createPendingForm() {
    const el = this._createEl('div', 'speech-form speech-form--chalie');
    const dots = this._createEl('div', 'thinking-indicator');
    for (let i = 0; i < 3; i++) {
      dots.appendChild(this._createEl('div', 'thinking-indicator__dot'));
    }
    el.appendChild(dots);
    this._spine.appendChild(el);
    this._scrollToBottom();
    return el;
  }

  /**
   * Replace a pending form's thinking dots with actual content.
   * @param {HTMLElement} form
   * @param {string} text
   * @param {{topic?: string, duration_ms?: number, removed_by?: string, removes?: string}} [meta]
   */
  resolvePendingForm(form, text, meta = {}) {
    form.innerHTML = '';
    const textEl = this._createEl('div', 'speech-form__text');
    textEl.textContent = text;
    form.appendChild(textEl);

    const metaRow = this._buildMetaRow(text, meta);
    form.appendChild(metaRow);

    // Handle removed_by parameter — mark for pending removal
    if (meta.removed_by) {
      form.classList.add('speech-form--pending-removal');
      this._messagesByRemovalId.set(meta.removed_by, form);
    } else {
      // If this form was marked for removal but no longer is, remove the class
      form.classList.remove('speech-form--pending-removal');
    }

    // Handle removes parameter — delete messages with matching removed_by
    if (meta.removes) {
      const removedEl = this._messagesByRemovalId.get(meta.removes);
      if (removedEl) {
        removedEl.remove();
        this._messagesByRemovalId.delete(meta.removes);
      }
    }

    this._setActiveForm(form);
    this._scrollToBottom();
  }

  /**
   * Replace a pending form with an error message.
   * @param {HTMLElement} form
   * @param {string} message
   */
  resolvePendingFormError(form, message) {
    form.innerHTML = '';
    form.classList.add('speech-form--error');
    const textEl = this._createEl('div', 'speech-form__text');
    textEl.textContent = message;
    form.appendChild(textEl);
    this._scrollToBottom();
  }

  /** Insert a capability card element into the spine. */
  insertCard(cardElement) {
    this._spine.appendChild(cardElement);
    this._scrollToBottom();
  }

  /** Append a pre-built tool result card element. */
  appendToolCard(cardElement) {
    this._spine.appendChild(cardElement);
    this._scrollToBottom();
    return cardElement;
  }

  /**
   * Register a pending form under a removal ID immediately, before onDone fires.
   * Allows the drift stream to find and remove the placeholder even when a
   * tool follow-up arrives before resolvePendingForm has been called.
   * @param {string} id — the removed_by token
   * @param {HTMLElement} el — the pending form element
   */
  registerPendingRemoval(id, el) {
    el.classList.add('speech-form--pending-removal');
    this._messagesByRemovalId.set(id, el);
  }

  /** Remove all children from the spine. */
  clear() {
    this._spine.innerHTML = '';
  }

  // ---------------------------------------------------------------------------
  // Private helpers
  // ---------------------------------------------------------------------------

  _buildMetaRow(text, meta) {
    const metaRow = this._createEl('div', 'speech-form__meta');

    // Timestamp — always shown
    const timestampEl = this._createEl('span', 'speech-form__timestamp');
    timestampEl.textContent = this._formatTimestamp(meta.ts ?? null);
    metaRow.appendChild(timestampEl);

    // TTS speak button — only shown when TTS is configured
    if (this._ttsEnabled) {
      const speakBtn = this._createEl('button', 'speech-form__speak-btn');
      speakBtn.setAttribute('aria-label', 'Read aloud');
      speakBtn.innerHTML = SPEAK_ICON;
      speakBtn.addEventListener('click', () => {
        document.dispatchEvent(new CustomEvent('chalie:speak', { detail: { text } }));
      });
      metaRow.appendChild(speakBtn);
    }

    return metaRow;
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
