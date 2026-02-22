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

    // Body (compiled HTML from backend, already sanitized)
    const body = document.createElement('div');
    body.className = 'tool-result-card__body';
    body.innerHTML = data.html;

    // Wire up YouTube click-to-play (onclick stripped by HTML sanitizer)
    const thumb = body.querySelector('.yt-th');
    const frame = body.querySelector('.yt-fr');
    if (thumb && frame) {
      thumb.style.cursor = 'pointer';
      thumb.addEventListener('click', () => {
        frame.src = frame.dataset.src;
        thumb.style.display = 'none';
        frame.style.display = 'block';
      });
    }

    card.appendChild(body);

    return card;
  }
}
