/**
 * Tool Result Card — renders backend-compiled card data in the conversation spine.
 *
 * Generic interactive conventions (any tool can use these):
 *
 *   Lazy-load embed
 *     Wrap a thumbnail and a media element in <[data-lazy-embed]>.
 *     Mark the clickable thumbnail with [data-lazy-thumb].
 *     Put the real URL in [data-lazy-src] on the media element (iframe/video/img).
 *     On click the thumbnail hides and the media element's src is set.
 *
 *   Carousel
 *     Mark the container with [data-carousel].
 *     Each slide:  [data-slide]
 *     Prev button: [data-prev]
 *     Next button: [data-next]
 *     Dot indicators: [data-dot]
 *     First slide must be visible (display:flex/block), rest display:none.
 *     Supports click navigation, dot navigation, and pointer drag/swipe.
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
    if (data.css && !document.getElementById(`card-style-${scopeId}`)) {
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

    // --- Generic lazy-load embeds ---
    // Tools declare: <[data-lazy-embed]> > <[data-lazy-thumb]> + <[data-lazy-src]>
    body.querySelectorAll('[data-lazy-embed]').forEach(embed => {
      const thumb = embed.querySelector('[data-lazy-thumb]');
      const media = embed.querySelector('[data-lazy-src]');
      if (!thumb || !media) return;
      thumb.style.cursor = 'pointer';
      thumb.addEventListener('click', () => {
        media.src = media.dataset.lazySrc;
        thumb.style.display = 'none';
        media.style.display = 'block';
      });
    });

    // --- Generic carousels ---
    // Tools declare: [data-carousel] > [data-slide] + [data-prev] + [data-next] + [data-dot]
    body.querySelectorAll('[data-carousel]').forEach(carousel => {
      const slides = [...carousel.querySelectorAll('[data-slide]')];
      const dots   = [...carousel.querySelectorAll('[data-dot]')];
      const prev   = carousel.querySelector('[data-prev]');
      const next   = carousel.querySelector('[data-next]');
      const n      = slides.length;
      if (n <= 1) return;
      let cur = 0;

      function showSlide(idx) {
        slides[cur].style.display = 'none';
        if (dots[cur]) { dots[cur].style.background = 'rgba(255,255,255,0.25)'; dots[cur].style.transform = 'scale(1)'; }
        cur = ((idx % n) + n) % n;
        slides[cur].style.display = 'flex';
        if (dots[cur]) { dots[cur].style.background = '#8A5CFF'; dots[cur].style.transform = 'scale(1.2)'; }
      }

      if (prev) prev.addEventListener('click', () => showSlide(cur - 1));
      if (next) next.addEventListener('click', () => showSlide(cur + 1));
      dots.forEach((d, i) => d.addEventListener('click', () => showSlide(i)));

      // Arrow button hover effects
      [prev, next].filter(Boolean).forEach(btn => {
        btn.addEventListener('mouseenter', () => {
          btn.style.background = 'rgba(138,92,255,0.15)';
          btn.style.borderColor = 'rgba(138,92,255,0.3)';
          btn.style.color = '#8A5CFF';
        });
        btn.addEventListener('mouseleave', () => {
          btn.style.background = 'rgba(255,255,255,0.07)';
          btn.style.borderColor = 'rgba(255,255,255,0.12)';
          btn.style.color = 'rgba(234,230,242,0.7)';
        });
      });

      // Pointer drag/swipe support (mouse + touch)
      let dragStartX = 0;
      carousel.addEventListener('pointerdown', e => { dragStartX = e.clientX; });
      carousel.addEventListener('pointerup', e => {
        const dx = e.clientX - dragStartX;
        if (Math.abs(dx) > 40) showSlide(cur + (dx < 0 ? 1 : -1));
      });
    });

    card.appendChild(body);

    return card;
  }
}
