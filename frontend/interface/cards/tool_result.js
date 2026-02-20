/**
 * Tool Result Card — renders backend-compiled card data in the conversation spine.
 */

export class ToolResultCard {
  /**
   * Build a card DOM element from backend-compiled card data.
   *
   * @param {{html, css, scope_id, title, accent_color, background_color, tool_name}} data
   * @returns {HTMLElement}
   */
  build(data) {
    const scopeId = data.scope_id;

    // Inject scoped <style> into document head (deduped by scope_id)
    if (!document.getElementById(`card-style-${scopeId}`)) {
      const style = document.createElement('style');
      style.id = `card-style-${scopeId}`;
      style.textContent = data.css;
      document.head.appendChild(style);
    }

    // Create card container
    const card = document.createElement('div');
    card.className = 'tool-result-card';
    card.setAttribute('data-card-scope', scopeId);
    card.setAttribute('data-tool', data.tool_name);

    // Apply dynamic styles
    if (data.background_color) {
      card.style.backgroundColor = data.background_color;
    }
    if (data.accent_color) {
      card.style.borderLeftColor = data.accent_color;
    }

    // Header with title
    const header = document.createElement('div');
    header.className = 'tool-result-card__header';
    const titleSpan = document.createElement('span');
    titleSpan.className = 'tool-result-card__title';
    titleSpan.textContent = data.title;
    header.appendChild(titleSpan);
    card.appendChild(header);

    // Body (compiled HTML from backend, already sanitized)
    const body = document.createElement('div');
    body.className = 'tool-result-card__body';
    body.innerHTML = data.html;
    card.appendChild(body);

    return card;
  }
}
