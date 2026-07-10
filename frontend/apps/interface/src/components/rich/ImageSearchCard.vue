<script setup lang="ts">
import { computed, reactive } from 'vue';

/**
 * Image search payload shape from the backend.
 */
export interface ImageSearchPayload {
  query: string;
  images: Array<{
    url: string;
    thumb_url: string;
    title: string;
    source: string;
    verified: boolean;
  }>;
}

const props = defineProps<{
  payload: ImageSearchPayload;
  synthesis?: string;
}>();

/**
 * Protocol gate for open-web image URLs. Parse the candidate as an ABSOLUTE URL
 * (no base) and accept only an explicit `http:`/`https:` scheme. Base-less
 * `new URL()` throws on relative, scheme-less, and protocol-relative input, so
 * those are refused — unlike an anchor-probe (`a.href = url; a.protocol`), which
 * resolves them against the page origin and would let a relative path through as
 * a same-origin, cookie-bearing request. Also rejects `javascript:`/`data:`/
 * `vbscript:`/`file:` via the allowlist. This is the security precedent for every
 * remote-media rich card — keep it strict; payload URLs are unsanitised.
 */
const HTTP_SCHEMES = new Set(['http:', 'https:']);
function isHttp(url: string): boolean {
  if (typeof url !== 'string' || !url.trim()) return false;
  try {
    return HTTP_SCHEMES.has(new URL(url).protocol);
  } catch {
    return false;
  }
}

interface ResolvedImage {
  thumbSrc: string;
  linkHref: string;
  title: string;
  source: string;
  verified: boolean;
}

const MAX_IMAGES = 5;

const resolved = computed<ResolvedImage[]>(() => {
  const raw = props.payload.images ?? [];
  const out: ResolvedImage[] = [];
  for (const img of raw.slice(0, MAX_IMAGES)) {
    const thumb = img.thumb_url && isHttp(img.thumb_url) ? img.thumb_url : '';
    const fallback = img.url && isHttp(img.url) ? img.url : '';
    const src = thumb || fallback;
    if (!src) continue;
    const href = isHttp(img.url) ? img.url : '';
    out.push({
      thumbSrc: src,
      linkHref: href,
      title: img.title || '',
      source: img.source || '',
      verified: Boolean(img.verified),
    });
  }
  return out;
});

/** Grid indices whose thumbnail 404'd at load time — swapped for the `⊘`
 * placeholder while keeping the cell's aspect box so layout stays intact. */
const failed = reactive(new Set<number>());
</script>

<template>
  <div v-if="resolved.length" class="rich-card image-card">
    <div class="image-card__header">
      <span class="image-card__query">{{ payload.query }}</span>
      <span v-if="synthesis" class="image-card__synthesis">{{ synthesis }}</span>
    </div>

    <div class="image-card__grid">
      <div v-for="(img, idx) in resolved" :key="idx" class="image-card__cell">
        <component
          :is="img.linkHref ? 'a' : 'div'"
          class="image-card__link"
          v-bind="img.linkHref ? { href: img.linkHref, target: '_blank', rel: 'noopener noreferrer' } : {}"
        >
          <img
            v-if="!failed.has(idx)"
            :src="img.thumbSrc"
            :alt="img.title || 'Image'"
            class="image-card__img"
            loading="lazy"
            @error="failed.add(idx)"
          />
          <div v-else class="image-card__img image-card__img--broken" aria-hidden="true">⊘</div>
        </component>
        <div v-if="img.verified" class="image-card__badge" role="img" aria-label="Verified">✓</div>
        <div class="image-card__meta">
          <span v-if="img.title" class="image-card__title">{{ img.title }}</span>
          <span v-if="img.source" class="image-card__source">{{ img.source }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped lang="scss">
/* Card-specific rules only; base .rich-card chrome lives globally in rich-card-base.scss. */

.rich-card.image-card {
  max-width: 620px;
}

.image-card__header {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 6px 12px;
  margin-bottom: 12px;
  padding-bottom: 10px;
  border-bottom: 1px solid var(--border);
}

.image-card__query {
  font-size: 0.92rem;
  font-weight: 600;
  color: var(--text-primary);
  letter-spacing: -0.01em;
}

.image-card__synthesis {
  font-size: 0.78rem;
  color: var(--text-tertiary);
  font-style: italic;
  margin-left: auto;
}

.image-card__grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
  gap: 10px;
}

.image-card__cell {
  position: relative;
  border-radius: 10px;
  overflow: hidden;
  border: 1px solid var(--border);
  background: var(--bg-input);
  transition: border-color 160ms ease;

  &:hover {
    border-color: var(--border-strong);
  }
}

.image-card__link {
  display: block;
  text-decoration: none;
  color: inherit;
}

.image-card__img {
  display: block;
  width: 100%;
  aspect-ratio: 4 / 3;
  object-fit: cover;
  background: color-mix(in oklab, var(--violet) 6%, var(--bg-2));
}

.image-card__img--broken {
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--text-tertiary);
  font-size: 1.4rem;
}

.image-card__badge {
  position: absolute;
  top: 5px;
  right: 5px;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: color-mix(in oklab, var(--violet) 70%, #fff);
  color: #fff;
  font-size: 0.6rem;
  display: flex;
  align-items: center;
  justify-content: center;
  pointer-events: none;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.25);
}

.image-card__meta {
  padding: 7px 8px 8px;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.image-card__title {
  font-size: 0.76rem;
  font-weight: 500;
  color: var(--text-primary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  line-height: 1.25;
  /* title/source are third-party search strings — isolate bidi control chars
     so a hostile RTL override can't visually reorder the attribution label. */
  unicode-bidi: isolate;
}

.image-card__source {
  font-size: 0.68rem;
  color: var(--text-tertiary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-family: var(--font-mono, 'JetBrains Mono', ui-monospace, monospace);
  unicode-bidi: isolate;
}
</style>
