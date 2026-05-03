/**
 * Article card module — shared renderer for News + Search.
 *
 * Implements the rich-media card contract: exports render(payload, synthesis, root).
 * Both news and search use the same layout: optional image thumbnail on the right,
 * a single synthesis paragraph, and a row of source pills at the bottom.
 */

const SOURCE_COLORS = [
  '#4ea2ff', '#FF2FD1', '#F2C94C', '#00F0FF', '#34d399',
  '#fb7185', '#8A5CFF', '#fbbf24', '#6E3FE6', '#B07CFF',
];

function pickColor(name) {
  let hash = 0;
  for (let i = 0; i < name.length; i++) {
    hash = ((hash << 5) - hash + name.charCodeAt(i)) | 0;
  }
  return SOURCE_COLORS[Math.abs(hash) % SOURCE_COLORS.length];
}

function initials(name) {
  return name
    .replace(/^https?:\/\/(www\.)?/, '')
    .replace(/\.com|\.org|\.net|\.io|\.eu|\.co\.uk/g, '')
    .slice(0, 2)
    .toUpperCase();
}

function extractDomain(url) {
  try {
    return new URL(url).hostname.replace(/^www\./, '');
  } catch {
    return url;
  }
}

/**
 * Build and mount the article card DOM into root.
 *
 * @param {object}      payload   - Tool result data
 * @param {string}      synthesis - LLM synthesis text
 * @param {HTMLElement}  root     - Container element
 */
export function render(payload, synthesis, root) {
  const card = document.createElement('div');
  const results = payload.results || [];
  const imageUrl = payload.image_url || null;
  card.className = 'rich-card article-card' + (imageUrl ? '' : ' article-card--no-image');

  if (imageUrl) {
    const thumb = document.createElement('div');
    thumb.className = 'article-card__thumb article-card__thumb--has-img';
    const img = document.createElement('img');
    img.src = imageUrl;
    img.alt = '';
    img.loading = 'lazy';
    img.onerror = () => { thumb.remove(); card.classList.add('article-card--no-image'); };
    thumb.appendChild(img);
    card.appendChild(thumb);
  }

  const body = document.createElement('div');
  body.className = 'article-card__body';

  const dek = document.createElement('p');
  dek.className = 'article-card__dek';
  dek.textContent = synthesis || '';
  body.appendChild(dek);

  if (results.length > 0) {
    const sources = document.createElement('div');
    sources.className = 'article-card__sources';

    const shown = results.slice(0, 4);
    const remaining = results.length - shown.length;

    for (const r of shown) {
      const domain = r.url ? extractDomain(r.url) : (r.source || '');
      const a = document.createElement('a');
      a.className = 'article-card__src';
      if (r.url) {
        a.href = r.url;
        a.target = '_blank';
        a.rel = 'noopener noreferrer';
      }

      const ico = document.createElement('span');
      ico.className = 'article-card__src-ico';
      ico.style.background = pickColor(domain);
      ico.textContent = initials(domain);
      a.appendChild(ico);

      a.appendChild(document.createTextNode(domain));
      sources.appendChild(a);
    }

    if (remaining > 0) {
      const meta = document.createElement('span');
      meta.className = 'article-card__meta';
      meta.textContent = `+ ${remaining} more`;
      sources.appendChild(meta);
    }

    body.appendChild(sources);
  }

  card.appendChild(body);
  root.appendChild(card);
}
