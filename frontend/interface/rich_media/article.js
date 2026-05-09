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
    hash = Math.trunc((hash << 5) - hash + (name.codePointAt(i) ?? 0));
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
  card.className = 'rich-card article-card';

  const body = document.createElement('div');
  body.className = 'article-card__body';

  const dek = document.createElement('p');
  dek.className = 'article-card__dek';
  dek.textContent = synthesis || '';
  body.appendChild(dek);

  if (imageUrl) {
    const img = document.createElement('img');
    img.className = 'article-card__img';
    img.src = imageUrl;
    img.alt = '';
    img.loading = 'lazy';
    img.onerror = () => img.remove();
    body.appendChild(img);
  }

  const domains = [];
  const seen = new Set();
  for (const r of results) {
    const d = r.url ? extractDomain(r.url) : (r.source || '');
    if (!d || seen.has(d)) continue;
    seen.add(d);
    domains.push(d);
  }

  if (domains.length > 0) {
    const sources = document.createElement('div');
    sources.className = 'article-card__sources';

    const shown = domains.slice(0, 4);
    const remaining = domains.length - shown.length;

    for (const domain of shown) {
      const pill = document.createElement('span');
      pill.className = 'article-card__src';

      const ico = document.createElement('span');
      ico.className = 'article-card__src-ico';
      ico.style.background = pickColor(domain);
      ico.textContent = initials(domain);
      pill.appendChild(ico);

      pill.appendChild(document.createTextNode(domain));
      sources.appendChild(pill);
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
