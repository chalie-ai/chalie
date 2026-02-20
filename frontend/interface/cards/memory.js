/**
 * Memory Context card.
 * Fetches GET /memory/context on demand.
 */
import { BaseCard } from './base.js';

export class MemoryCard extends BaseCard {
  constructor(api, renderer) {
    super(api, renderer);
  }

  async fetch() {
    try {
      const data = await this._api.getMemoryContext();
      this.hide();

      const bodyHTML = this.template(data);
      this.render('Memory', bodyHTML);
      this.show();
    } catch (err) {
      console.debug('Memory fetch failed:', err);
    }
  }

  template(data) {
    let html = '';

    // Traits summary
    if (data.traits_summary) {
      html += `<div class="memory-section">
        <div class="memory-section__title">Traits</div>
        <div>${this._escapeHtml(data.traits_summary)}</div>
      </div>`;
    }

    // Facts
    if (data.facts && data.facts.length > 0) {
      const factsHtml = data.facts.map(f =>
        `<div class="memory-fact">
          <span class="memory-fact__key">${this._escapeHtml(f.key)}</span>
          <span>
            <span class="memory-fact__value">${this._escapeHtml(f.value)}</span>
            <span class="memory-fact__confidence">${Math.round((f.confidence || 0) * 100)}%</span>
          </span>
        </div>`
      ).join('');

      html += `<div class="memory-section">
        <div class="memory-section__title">Known Facts</div>
        ${factsHtml}
      </div>`;
    }

    // Significant episodes
    if (data.significant_episodes && data.significant_episodes.length > 0) {
      const episodesHtml = data.significant_episodes.map(ep =>
        `<div class="memory-fact">
          <span>${this._escapeHtml(ep.gist)}</span>
        </div>`
      ).join('');

      html += `<div class="memory-section">
        <div class="memory-section__title">Significant Episodes</div>
        ${episodesHtml}
      </div>`;
    }

    // Active concepts
    if (data.concepts && data.concepts.length > 0) {
      const conceptsHtml = data.concepts.map(c =>
        `<div class="memory-fact">
          <span class="memory-fact__key">${this._escapeHtml(c.name)}</span>
          <span class="memory-fact__value">${this._escapeHtml(c.definition || '')}</span>
        </div>`
      ).join('');

      html += `<div class="memory-section">
        <div class="memory-section__title">Active Concepts</div>
        ${conceptsHtml}
      </div>`;
    }

    return html || '<div>No memory context available.</div>';
  }

  _escapeHtml(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }
}
