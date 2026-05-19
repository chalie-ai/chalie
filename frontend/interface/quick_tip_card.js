/**
 * Quick Tip card — slide-up informational card for feature discovery.
 *
 * Periodic feature hints that surface above the input dock. Each card shows:
 *   - Feature icon + "Quick tip" label
 *   - One-sentence body describing the feature
 *   - "Try saying" example prompt
 *   - Dismiss ✕ button
 *   - "Don't show more tips" mute button
 *
 * Visual design sourced from Claude Design prototype (section 10, Tool Widgets v2).
 * Matches Radiant design system — violet accent, frosted glass, slide-up animation.
 */

export class QuickTipCard {
  constructor({ getHost }) {
    this._getHost = getHost;
    this._el = null;
    this._visible = false;
  }

  /**
   * Call once after the DOM is ready to mount the card shell.
   */
  init() {
    this._el = this._buildCard();
    document.body.appendChild(this._el);
  }

  /**
   * Handle a quick_tip event from the backend.
   * @param {{ tip_id: string, title: string, body: string, example: string, icon?: string, url?: string }} data
   */
  handleTip(data) {
    if (!data || !data.tip_id) return;
    if (this._visible) return;               // don't stack tips

    this._render(data);
    this._visible = true;
    this._el.setAttribute('aria-hidden', 'false');
    requestAnimationFrame(() => this._el.classList.add('tip--visible'));
  }

  // ---------------------------------------------------------------------------
  // Internal
  // ---------------------------------------------------------------------------

  _buildCard() {
    const card = document.createElement('div');
    card.className = 'tip';
    card.setAttribute('role', 'status');
    card.setAttribute('aria-live', 'polite');
    card.setAttribute('aria-hidden', 'true');

    card.innerHTML = `
      <button class="tip__dismiss" aria-label="Dismiss tip">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
             stroke-width="2.4" stroke-linecap="round" aria-hidden="true">
          <line x1="18" y1="6" x2="6" y2="18"/>
          <line x1="6" y1="6" x2="18" y2="18"/>
        </svg>
      </button>
      <div class="tip__head">
        <div class="tip__icon"></div>
        <div class="tip__label">Quick tip</div>
      </div>
      <p class="tip__body"></p>
      <div class="tip__example">
        <div class="tip__example-label">Try saying</div>
        <div class="tip__example-prompt"></div>
      </div>
      <div class="tip__foot">
        <button class="tip__mute">Don’t show more tips</button>
      </div>
    `;

    // Dismiss ✕
    card.querySelector('.tip__dismiss').addEventListener('click', () => {
      this._dismiss();
    });

    // "Don't show more tips" — dismiss + tell backend to disable
    card.querySelector('.tip__mute').addEventListener('click', () => {
      this._mute();
    });

    // Esc key
    this._onKeydown = (e) => {
      if (e.key === 'Escape' && this._visible) this._dismiss();
    };
    document.addEventListener('keydown', this._onKeydown);

    return card;
  }

  /**
   * Populate card content from tip data.
   */
  _render(data) {
    const iconEl = this._el.querySelector('.tip__icon');
    const bodyEl = this._el.querySelector('.tip__body');
    const promptEl = this._el.querySelector('.tip__example-prompt');

    // Icon — use SVG from data.icon_svg if provided, else fallback to lightbulb
    iconEl.innerHTML = data.icon_svg || this._fallbackIcon();
    bodyEl.textContent = data.body || '';
    promptEl.textContent = data.example || '';

    // Store tip_id for dismiss/mute calls
    this._el.dataset.tipId = data.tip_id;
  }

  /**
   * Dismiss the current tip (✕ button or Esc).
   * Sends dismiss to backend so this specific tip is marked.
   */
  _dismiss() {
    if (!this._visible) return;
    const tipId = this._el.dataset.tipId;
    this._close();

    if (tipId) {
      const base = this._getHost() ? this._getHost().replace(/\/$/, '') : '';
      fetch(`${base}/api/tips/dismiss`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },

        body: JSON.stringify({ tip_id: tipId }),
      }).catch((e) => console.warn('[QuickTip] dismiss failed:', e));
    }
  }

  /**
   * "Don't show more tips" — disable the feature entirely.
   * Sends mute to backend which sets the quick_tips_enabled setting to false.
   */
  _mute() {
    if (!this._visible) return;
    this._close();

    const base = this._getHost() ? this._getHost().replace(/\/$/, '') : '';
    fetch(`${base}/api/tips/mute`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
    }).catch((e) => console.warn('[QuickTip] mute failed:', e));
  }

  _close() {
    this._visible = false;
    this._el.classList.remove('tip--visible');
    this._el.classList.add('tip--leaving');
    this._el.setAttribute('aria-hidden', 'true');

    const cleanup = () => {
      this._el.classList.remove('tip--leaving');
    };
    this._el.addEventListener('transitionend', cleanup, { once: true });
    setTimeout(cleanup, 400);  // safety fallback
  }

  _fallbackIcon() {
    return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
      stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <path d="M9 18h6"/><path d="M10 22h4"/>
      <path d="M12 2a7 7 0 0 0-4 12.7V17h8v-2.3A7 7 0 0 0 12 2z"/>
    </svg>`;
  }
}
