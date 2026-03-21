/**
 * Conversation spine DOM renderer.
 */
import { BlockRenderer } from './blocks.js';

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
    this._blockRenderer = new BlockRenderer();
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
   * @param {Array} blocks — block protocol array
   * @param {{topic?: string, duration_ms?: number}} [meta]
   * @param {{inWorkingMemory?: boolean}} [options]
   */
  appendChalieForm(blocks, meta = {}, { inWorkingMemory = true } = {}) {
    const el = this._createEl('div', 'speech-form speech-form--chalie');
    if (!inWorkingMemory) el.classList.add('message--faded');
    const textEl = this._createEl('div', 'speech-form__text');
    textEl.appendChild(this._blockRenderer.render(blocks));
    el.appendChild(textEl);

    const metaRow = this._buildMetaRow(this._extractPlainText(blocks), meta);
    el.appendChild(metaRow);

    // Collapse any narration bubbles before appending the response
    this._collapseNarrations();

    this._spine.appendChild(el);
    this._setActiveForm(el);
    this._scrollToBottom();
    return el;
  }

  /**
   * Prepend a Chalie speech form (for scroll-up pagination).
   * @param {Array} blocks — block protocol array
   * @param {{topic?: string, duration_ms?: number}} [meta]
   * @param {{inWorkingMemory?: boolean}} [options]
   */
  prependChalieForm(blocks, meta = {}, { inWorkingMemory = true } = {}) {
    const el = this._createEl('div', 'speech-form speech-form--chalie');
    if (!inWorkingMemory) el.classList.add('message--faded');
    const textEl = this._createEl('div', 'speech-form__text');
    textEl.appendChild(this._blockRenderer.render(blocks));
    el.appendChild(textEl);

    const metaRow = this._buildMetaRow(this._extractPlainText(blocks), meta);
    el.appendChild(metaRow);

    this._spine.prepend(el);
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
   * Append a narration bubble — lightweight progress indicator during ACT loops.
   * @param {string} text — narration line from the LLM
   * @param {number} step — iteration number
   */
  appendNarrationBubble(text, step) {
    const bubble = this._createEl('div', 'narration-bubble');
    bubble.dataset.step = step;
    bubble.textContent = text;

    // Insert before the pending form (thinking dots) if present
    const pending = this._spine.querySelector('.thinking-indicator')?.parentElement;
    if (pending) {
      this._spine.insertBefore(bubble, pending);
    } else {
      this._spine.appendChild(bubble);
    }
    this._scrollToBottom();
    return bubble;
  }

  /**
   * Append a steer bubble — user's redirect shown inline with narration.
   * @param {string} text — the user's steering message
   */
  appendSteerBubble(text) {
    const bubble = this._createEl('div', 'steer-bubble');
    bubble.textContent = text;

    const pending = this._spine.querySelector('.thinking-indicator')?.parentElement;
    if (pending) {
      this._spine.insertBefore(bubble, pending);
    } else {
      this._spine.appendChild(bubble);
    }
    this._scrollToBottom();
    return bubble;
  }

  /**
   * Replace a pending form's thinking dots with actual content.
   * @param {HTMLElement} form
   * @param {Array} blocks — block protocol array
   * @param {{topic?: string, duration_ms?: number}} [meta]
   */
  resolvePendingForm(form, blocks, meta = {}) {
    form.innerHTML = '';
    const textEl = this._createEl('div', 'speech-form__text');
    textEl.appendChild(this._blockRenderer.render(blocks));
    form.appendChild(textEl);

    const metaRow = this._buildMetaRow(this._extractPlainText(blocks), meta);
    form.appendChild(metaRow);

    // Collapse narration bubbles into an expandable group
    this._collapseNarrations();

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

  /**
   * Upgrade a pending (thinking dots) form to a brief placeholder phrase.
   * Called after 2 seconds when the response is still in flight.
   * @param {HTMLElement} form
   */
  upgradePendingText(form) {
    const phrases = ['Working on it...', 'One moment...', 'On it...', 'Thinking...'];
    const text = phrases[Math.floor(Math.random() * phrases.length)];
    const dots = form.querySelector('.thinking-indicator');
    if (!dots) return; // already resolved
    form.innerHTML = '';
    const textEl = this._createEl('div', 'speech-form__text');
    textEl.textContent = text;
    textEl.style.opacity = '0.80';
    form.appendChild(textEl);
  }

  /** Remove all children from the spine. */
  clear() {
    this._spine.innerHTML = '';
  }

  // ---------------------------------------------------------------------------
  // Private helpers
  // ---------------------------------------------------------------------------

  _collapseNarrations() {
    const bubbles = this._spine.querySelectorAll('.narration-bubble:not(.narration--collapsed), .steer-bubble:not(.narration--collapsed)');
    if (bubbles.length < 2) return; // Don't collapse a single bubble

    const wrapper = this._createEl('div', 'narration-group narration-group--collapsed');
    const toggle = this._createEl('button', 'narration-toggle');
    toggle.textContent = `${bubbles.length} steps`;
    toggle.addEventListener('click', () => {
      wrapper.classList.toggle('narration-group--collapsed');
      toggle.textContent = wrapper.classList.contains('narration-group--collapsed')
        ? `${bubbles.length} steps`
        : 'collapse';
    });
    wrapper.appendChild(toggle);

    // Move bubbles into the wrapper
    const firstBubble = bubbles[0];
    firstBubble.parentNode.insertBefore(wrapper, firstBubble);
    for (const b of bubbles) {
      b.classList.add('narration--collapsed');
      wrapper.appendChild(b);
    }
  }

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

    // Speak button — enters voice mode and plays this message
    if (text) {
      const speakBtn = this._createEl('button', 'speech-form__speak-btn');
      speakBtn.setAttribute('aria-label', 'Listen to this message');
      speakBtn.innerHTML = SPEAK_ICON;
      speakBtn.addEventListener('click', () => {
        document.dispatchEvent(new CustomEvent('chalie:speak-message', {
          detail: { text }
        }));
      });
      metaRow.appendChild(speakBtn);
    }

    return metaRow;
  }

  _extractPlainText(blocks) {
    if (!Array.isArray(blocks)) return '';
    const parts = [];
    for (const b of blocks) {
      if (b.type === 'text' || b.type === 'header' || b.type === 'code') {
        if (b.content) parts.push(b.content);
      } else if (b.type === 'list' && Array.isArray(b.items)) {
        parts.push(b.items.join('\n'));
      } else if (b.type === 'keyvalue' && Array.isArray(b.pairs)) {
        parts.push(b.pairs.map(p => `${p.key}: ${p.value}`).join('\n'));
      } else if (b.type === 'table') {
        if (Array.isArray(b.rows)) parts.push(b.rows.map(r => r.join(', ')).join('\n'));
      } else if (b.type === 'alert' && b.message) {
        parts.push(b.message);
      }
    }
    return parts.join('\n');
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
   * Uses instant behavior so it can't race with content appends.
   */
  forceScrollToBottom() {
    this._userScrolledUp = false;
    window.scrollTo({ top: document.body.scrollHeight, behavior: 'instant' });
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
