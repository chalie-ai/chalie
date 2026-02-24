/**
 * Moment Card — client-side card for displaying pinned moments.
 *
 * Used by both the search overlay results and drift-stream card events.
 * Matches the backend MomentCardService structure.
 */
export class MomentCard {
  /**
   * @param {Object} data — moment data from the API
   */
  constructor(data) {
    this._data = data;
  }

  /**
   * Build and return a DOM element for this moment.
   * @returns {HTMLElement}
   */
  build() {
    const card = document.createElement('div');
    card.className = 'moment-card';
    card.dataset.momentId = this._data.id || '';

    // 1. Title
    const titleEl = document.createElement('div');
    titleEl.className = 'moment-card__title';
    titleEl.textContent = this._data.title || 'Moment';
    card.appendChild(titleEl);

    // 2. Pinned message (quoted block)
    const messageEl = document.createElement('div');
    messageEl.className = 'moment-card__message';
    messageEl.textContent = this._data.message_text || '';
    card.appendChild(messageEl);

    // 3. Summary
    if (this._data.summary) {
      const summaryEl = document.createElement('div');
      summaryEl.className = 'moment-card__summary';
      summaryEl.textContent = this._data.summary;
      card.appendChild(summaryEl);
    }

    // 4. Gists
    const gists = this._data.gists || [];
    if (gists.length > 0) {
      const gistsEl = document.createElement('div');
      gistsEl.className = 'moment-card__gists';
      for (const gist of gists.slice(0, 4)) {
        const item = document.createElement('div');
        item.className = 'moment-card__gist-item';
        item.textContent = gist;
        gistsEl.appendChild(item);
      }
      card.appendChild(gistsEl);
    }

    // 5. Pinned time
    if (this._data.pinned_at) {
      const footerEl = document.createElement('div');
      footerEl.className = 'moment-card__footer';
      footerEl.textContent = `Pinned ${this._formatDate(this._data.pinned_at)}`;
      card.appendChild(footerEl);
    }

    return card;
  }

  _formatDate(dateStr) {
    try {
      const d = new Date(dateStr);
      const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
      const day = String(d.getDate()).padStart(2, '0');
      return `${day} ${months[d.getMonth()]} ${d.getFullYear()}, ${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}`;
    } catch {
      return dateStr;
    }
  }
}
