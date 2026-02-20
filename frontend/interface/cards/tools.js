/**
 * Tools Card — inline spine summary of connected integrations.
 */
import { BaseCard } from './base.js';

export class ToolsCard extends BaseCard {
  /**
   * @param {import('../api.js').ApiClient} api
   * @param {import('../renderer.js').Renderer} renderer
   */
  constructor(api, renderer) {
    super(api, renderer);
  }

  async fetch() {
    try {
      const data = await this._api.getTools();
      this.hide();

      const bodyHTML = this.template(data);
      this.render('Integrations', bodyHTML);
      this.show();
    } catch (err) {
      console.debug('ToolsCard fetch failed:', err);
    }
  }

  template(data) {
    const tools = data.tools || [];
    const connected = tools.filter(t => t.status === 'connected');
    const available = tools.filter(t => t.status === 'available');

    // Icon chips
    let chipsHtml = '';
    if (connected.length > 0) {
      const chips = connected.map(t => {
        const iconHtml = t.icon ? this._renderIconHtml(t.icon) : this._esc(t.name.charAt(0).toUpperCase());
        return `<span class="tools-card-chip" title="${this._esc(t.display_name || t.name)}">${iconHtml}</span>`;
      }).join('');
      chipsHtml = `<div class="tools-card-chips">${chips}</div>`;
    }

    // Summary text
    let summaryText;
    if (connected.length === 0) {
      summaryText = 'No integrations connected yet.';
    } else if (connected.length === 1) {
      summaryText = `${this._esc(connected[0].display_name || connected[0].name)} is active.`;
    } else {
      const names = connected.map(t => this._esc(t.display_name || t.name));
      const last = names.pop();
      summaryText = `${names.join(', ')} and ${last} are active.`;
    }

    // Available note
    const availableNote = available.length > 0
      ? `<div class="tools-card-available">${available.length} available to connect</div>`
      : '';

    return `
      ${chipsHtml}
      <div class="tools-card-summary">
        <span class="tools-card-text">${summaryText}</span>
      </div>
      ${availableNote}
    `;
  }

  _esc(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  _renderIconHtml(icon) {
    // External URL
    if (icon.startsWith('http://') || icon.startsWith('https://') || icon.startsWith('/')) {
      return `<img src="${this._esc(icon)}" style="width: 100%; height: 100%; object-fit: contain;" alt="icon">`;
    }

    // FontAwesome icon
    return `<i class="fa-solid ${this._esc(icon)}"></i>`;
  }
}
