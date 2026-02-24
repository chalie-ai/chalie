/**
 * Moment Search — full-screen recall overlay.
 *
 * Provides a dark overlay with a search input for semantic recall of
 * pinned moments. Results render as MomentCard instances.
 */
import { MomentCard } from './cards/moment.js';

export class MomentSearch {
  /**
   * @param {Function} apiFetch — function(path) that returns fetch Response
   */
  constructor(apiFetch) {
    this._apiFetch = apiFetch;
    this._dialog = null;
    this._input = null;
    this._results = null;
    this._debounceTimer = null;
    this._build();
  }

  open() {
    this._dialog.showModal();
    this._input.value = '';
    this._results.innerHTML = '';
    this._showEmpty();
    setTimeout(() => this._input.focus(), 100);
  }

  close() {
    this._dialog.close();
  }

  // ---------------------------------------------------------------------------
  // Private
  // ---------------------------------------------------------------------------

  _build() {
    this._dialog = document.createElement('dialog');
    this._dialog.className = 'moment-search-dialog';
    this._dialog.setAttribute('aria-label', 'Recall');

    this._dialog.innerHTML = `
      <div class="moment-search-dialog__content">
        <div class="moment-search-dialog__header">
          <h2 class="moment-search-dialog__title">Recall</h2>
          <button class="moment-search-dialog__close btn-icon" aria-label="Close">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <line x1="18" y1="6" x2="6" y2="18"></line>
              <line x1="6" y1="6" x2="18" y2="18"></line>
            </svg>
          </button>
        </div>
        <input type="text" class="moment-search-dialog__input"
               placeholder="Recall something..." autocomplete="off" />
        <div class="moment-search-dialog__results"></div>
      </div>
    `;

    document.body.appendChild(this._dialog);

    this._input = this._dialog.querySelector('.moment-search-dialog__input');
    this._results = this._dialog.querySelector('.moment-search-dialog__results');

    // Close button
    this._dialog.querySelector('.moment-search-dialog__close')
      .addEventListener('click', () => this.close());

    // Escape key
    this._dialog.addEventListener('cancel', (e) => {
      e.preventDefault();
      this.close();
    });

    // Debounced search
    this._input.addEventListener('input', () => {
      clearTimeout(this._debounceTimer);
      const query = this._input.value.trim();
      if (!query) {
        this._showEmpty();
        return;
      }
      this._showLoading();
      this._debounceTimer = setTimeout(() => this._search(query), 500);
    });
  }

  async _search(query) {
    try {
      const res = await this._apiFetch(`/moments/search?q=${encodeURIComponent(query)}`);
      if (!res.ok) throw new Error('Search failed');
      const data = await res.json();
      const items = data.items || [];

      this._results.innerHTML = '';

      if (items.length === 0) {
        this._results.innerHTML =
          '<div class="moment-search-dialog__empty">I couldn\'t recall anything like that yet.</div>';
        return;
      }

      for (const item of items) {
        const card = new MomentCard(item);
        const el = card.build();
        el.addEventListener('click', () => {
          document.dispatchEvent(new CustomEvent('chalie:show-moment', {
            detail: { moment: item }
          }));
          this.close();
        });
        el.style.cursor = 'pointer';
        this._results.appendChild(el);
      }
    } catch (err) {
      this._results.innerHTML =
        '<div class="moment-search-dialog__empty">Something went wrong. Try again.</div>';
    }
  }

  _showEmpty() {
    this._results.innerHTML =
      '<div class="moment-search-dialog__empty">Your remembered answers will appear here.</div>';
  }

  _showLoading() {
    this._results.innerHTML =
      '<div class="moment-search-dialog__shimmer"><div></div><div></div><div></div></div>';
  }
}
