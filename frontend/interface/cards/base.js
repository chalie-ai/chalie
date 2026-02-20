/**
 * Base class for capability cards.
 */
export class BaseCard {
  /**
   * @param {import('../api.js').ApiClient} api
   * @param {import('../renderer.js').Renderer} renderer
   */
  constructor(api, renderer) {
    this._api = api;
    this._renderer = renderer;
    this._element = null;
    this._dismissTimeout = null;
  }

  /**
   * Build the card DOM element.
   * @param {string} title — card header title
   * @param {string} bodyHTML — inner HTML for the card body
   * @returns {HTMLElement}
   */
  render(title, bodyHTML) {
    const card = document.createElement('div');
    card.className = 'capability-card';

    const header = document.createElement('div');
    header.className = 'capability-card__header';

    const titleEl = document.createElement('span');
    titleEl.className = 'capability-card__title';
    titleEl.textContent = title;

    const dismissBtn = document.createElement('button');
    dismissBtn.className = 'capability-card__dismiss';
    dismissBtn.setAttribute('aria-label', 'Dismiss');
    dismissBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <line x1="18" y1="6" x2="6" y2="18"></line>
      <line x1="6" y1="6" x2="18" y2="18"></line>
    </svg>`;
    dismissBtn.addEventListener('click', () => this.hide());

    header.appendChild(titleEl);
    header.appendChild(dismissBtn);

    const body = document.createElement('div');
    body.className = 'capability-card__body';
    body.innerHTML = bodyHTML;

    card.appendChild(header);
    card.appendChild(body);

    this._element = card;
    return card;
  }

  /** Insert the card into the conversation spine. */
  show() {
    if (this._element) {
      this._renderer.insertCard(this._element);
    }
  }

  /** Remove the card from the DOM. */
  hide() {
    if (this._dismissTimeout) {
      clearTimeout(this._dismissTimeout);
      this._dismissTimeout = null;
    }
    if (this._element && this._element.parentNode) {
      this._element.remove();
    }
    this._element = null;
  }

  /**
   * Auto-dismiss after a delay.
   * @param {number} ms
   */
  autoDismiss(ms) {
    this._dismissTimeout = setTimeout(() => this.hide(), ms);
  }

  /**
   * Override in subclasses: build body HTML from data.
   * @param {any} data
   * @returns {string}
   */
  template(data) {
    return '';
  }
}
