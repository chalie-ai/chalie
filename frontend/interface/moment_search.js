/**
 * Moment Search — full-screen recall overlay.
 *
 * Provides a dark overlay with a search input for semantic recall of
 * pinned moments.
 */

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

      this._results.textContent = '';

      if (items.length === 0) {
        const empty = document.createElement('div');
        empty.className = 'moment-search-dialog__empty';
        empty.textContent = 'I couldn\'t recall anything like that yet.';
        this._results.appendChild(empty);
        return;
      }

      for (const item of items) {
        const el = document.createElement('div');
        el.className = 'moment-search-dialog__item';
        const title = item.title || item.topic || 'Moment';
        const text = item.summary || item.message_text || '';

        const titleEl = document.createElement('div');
        titleEl.className = 'moment-search-dialog__item-title';
        titleEl.textContent = title;
        el.appendChild(titleEl);

        if (text) {
          const textEl = document.createElement('div');
          textEl.className = 'moment-search-dialog__item-text';
          textEl.textContent = text;
          el.appendChild(textEl);
        }

        el.style.cursor = 'pointer';
        el.addEventListener('click', () => this.close());
        this._results.appendChild(el);
      }
    } catch (err) {
      this._results.textContent = '';
      const errEl = document.createElement('div');
      errEl.className = 'moment-search-dialog__empty';
      errEl.textContent = 'Something went wrong. Try again.';
      this._results.appendChild(errEl);
    }
  }

  _showEmpty() {
    this._results.textContent = '';
    const empty = document.createElement('div');
    empty.className = 'moment-search-dialog__empty';
    empty.textContent = 'Your remembered answers will appear here.';
    this._results.appendChild(empty);
  }

  _showLoading() {
    this._results.textContent = '';
    const shimmer = document.createElement('div');
    shimmer.className = 'moment-search-dialog__shimmer';
    for (let i = 0; i < 3; i++) {
      shimmer.appendChild(document.createElement('div'));
    }
    this._results.appendChild(shimmer);
  }
}
