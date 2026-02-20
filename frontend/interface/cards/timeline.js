/**
 * Conversation Timeline card.
 * Fetches GET /conversation/summary and renders grouped timeline.
 */
import { BaseCard } from './base.js';

export class TimelineCard extends BaseCard {
  constructor(api, renderer) {
    super(api, renderer);
  }

  async fetch() {
    try {
      const data = await this._api.getConversationSummary();
      this.hide();

      const bodyHTML = this.template(data);
      this.render('Timeline', bodyHTML);
      this.show();

      // Bind collapse/expand after insertion
      this._bindSections();
    } catch (err) {
      console.debug('Timeline fetch failed:', err);
    }
  }

  template(data) {
    let html = '';

    if (data.today && data.today.length > 0) {
      html += this._renderSection('Today', data.today, true);
    }

    if (data.this_week && data.this_week.length > 0) {
      html += this._renderSection('This Week', data.this_week, false);
    }

    if (data.older_highlights && data.older_highlights.length > 0) {
      html += this._renderSection('Older Highlights', data.older_highlights, false);
    }

    return html || '<div>No conversation history.</div>';
  }

  _renderSection(title, items, expanded) {
    const expandedClass = expanded ? ' timeline-section--expanded' : '';
    const itemsHtml = items.map(item => {
      const time = item.timestamp
        ? `<span class="timeline-item__time">${this._formatTime(item.timestamp)}</span> `
        : '';
      return `<div class="timeline-item">
        ${time}${this._escapeHtml(item.content || item.gist || '')}
      </div>`;
    }).join('');

    return `<div class="timeline-section${expandedClass}" data-section>
      <div class="timeline-section__title">${this._escapeHtml(title)}</div>
      <div class="timeline-section__body">${itemsHtml}</div>
    </div>`;
  }

  _bindSections() {
    if (!this._element) return;
    const titles = this._element.querySelectorAll('.timeline-section__title');
    titles.forEach(titleEl => {
      titleEl.addEventListener('click', () => {
        const section = titleEl.closest('.timeline-section');
        section.classList.toggle('timeline-section--expanded');
      });
    });
  }

  _formatTime(timestamp) {
    try {
      const d = new Date(timestamp);
      return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } catch {
      return '';
    }
  }

  _escapeHtml(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }
}
