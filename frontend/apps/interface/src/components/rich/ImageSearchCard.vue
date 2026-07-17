<script setup lang="ts">
import { computed, reactive, ref, onMounted, onBeforeUnmount } from 'vue';
import { isAbsoluteHttpUrl } from '../../utils/url';

/**
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
  /** Part of the shared rich-card prop contract; the image summary is not shown here. */
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
 *
 * Now uses the shared `isAbsoluteHttpUrl()` from `../../utils/url`.
 */

/** Hostname of an absolute URL, `www.` stripped. Best-effort: falls back to the
 * raw string when parsing fails (only fed gated http(s) URLs, so it won't). */
function extractDomain(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, '');
  } catch {
    return url;
  }
}

/** Two-character initials for a source pill's icon. */
function initials(name: string): string {
  return name
    .replace(/^https?:\/\/(www\.)?/, '')
    .replace(/\.com|\.org|\.net|\.io|\.eu|\.co\.uk/g, '')
    .slice(0, 2)
    .toUpperCase();
}

/** Deterministic pill-icon colour hashed from the domain (stable per source). */
const SOURCE_COLORS: readonly string[] = [
  '#4ea2ff', '#FF2FD1', '#F2C94C', '#00F0FF', '#34d399',
  '#fb7185', '#8A5CFF', '#fbbf24', '#6E3FE6', '#B07CFF',
] as const;
function pickColor(name: string): string {
  let hash = 0;
  for (let i = 0; i < name.length; i++) {
    hash = Math.trunc((hash << 5) - hash + (name.codePointAt(i) ?? 0));
  }
  return SOURCE_COLORS[Math.abs(hash) % SOURCE_COLORS.length] as string;
}

interface ResolvedImage {
  thumbSrc: string;
  linkHref: string;
  title: string;
  domain: string;
  verified: boolean;
}

const MAX_IMAGES = 5;

const resolved = computed<ResolvedImage[]>(() => {
  const raw = props.payload.images ?? [];
  const out: ResolvedImage[] = [];
  for (const img of raw.slice(0, MAX_IMAGES)) {
    const thumb = img.thumb_url && isAbsoluteHttpUrl(img.thumb_url) ? img.thumb_url : '';
    const fallback = img.url && isAbsoluteHttpUrl(img.url) ? img.url : '';
    const src = thumb || fallback;
    if (!src) continue;
    const href = isAbsoluteHttpUrl(img.url) ? img.url : '';
    const domainSrc = img.source && isAbsoluteHttpUrl(img.source) ? img.source : href;
    out.push({
      thumbSrc: src,
      linkHref: href,
      title: img.title || '',
      domain: domainSrc ? extractDomain(domainSrc) : '',
      verified: Boolean(img.verified),
    });
  }
  return out;
});

/** Deduped source domains across the resolved images, in first-seen order —
 * folded into clean pills instead of a full URL under every thumbnail. */
const domains = computed<string[]>(() => {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const img of resolved.value) {
    if (!img.domain || seen.has(img.domain)) continue;
    seen.add(img.domain);
    out.push(img.domain);
  }
  return out;
});

/** Carousel indices whose thumbnail 404'd at load time — swapped for the `⊘`
 * placeholder while keeping the slide's aspect box so layout stays intact. */
const failed = reactive(new Set<number>());

/** Sources fold: collapsed to a "N sources" pill (same trace-pill affordance as
 * the ACT trail), expanded to the domain chips on click. */
const sourcesOpen = ref(false);

// ── Carousel navigation ─────────────────────────────────────────────────────
// Local view state only (no fetch/store/WS): a scroll-snap track paged by the
// ‹ / › buttons. Buttons appear only when the slides overflow, and each disables
// at its end of the track.
const track = ref<HTMLElement | null>(null);
const overflowing = ref(false);
const atStart = ref(true);
const atEnd = ref(false);
const isDragging = ref(false);
const GAP = 16;
const SWIPE_THRESHOLD = 45; // px of horizontal travel that reads as a swipe, not a tap

function updateNav(): void {
  const el = track.value;
  if (!el) return;
  const max = el.scrollWidth - el.clientWidth;
  overflowing.value = max > 1;
  atStart.value = el.scrollLeft <= 1;
  atEnd.value = el.scrollLeft >= max - 1;
}

function step(dir: 1 | -1): void {
  const el = track.value;
  if (!el) return;
  const cell = el.querySelector<HTMLElement>('.image-card__cell');
  const width = cell ? cell.getBoundingClientRect().width + GAP : el.clientWidth;
  el.scrollBy({ left: dir * width, behavior: 'smooth' });
}

// ── Swipe gesture ────────────────────────────────────────────────────────────
// Pointer Events (not touch-only) so a mouse-drag on desktop behaves the same
// as a touchscreen swipe. Only the release delta decides prev/next — a plain
// tap never runs the threshold math — and every path (arrows + swipe) funnels
// through the same step(), so bounds/wrap behaviour has one source of truth.
let dragging = false;
let dragPointerId: number | null = null;
let dragStartX = 0;
let dragStartY = 0;
let suppressClick = false;

function onTrackPointerDown(e: PointerEvent): void {
  if (!overflowing.value || (e.pointerType === 'mouse' && e.button !== 0)) return;
  dragging = true;
  isDragging.value = true;
  dragPointerId = e.pointerId;
  dragStartX = e.clientX;
  dragStartY = e.clientY;
  (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
}

function onTrackPointerMove(e: PointerEvent): void {
  if (!dragging || e.pointerId !== dragPointerId) return;
  // Own the gesture outright once dragging — blocks the native image/link
  // drag-ghost and any incidental text selection a mouse-drag would start.
  e.preventDefault();
}

function endDrag(e: PointerEvent, commit: boolean): void {
  if (!dragging || e.pointerId !== dragPointerId) return;
  dragging = false;
  isDragging.value = false;
  dragPointerId = null;
  if (!commit) return; // pointercancel (e.g. the browser took over for a vertical scroll)
  const dx = e.clientX - dragStartX;
  const dy = e.clientY - dragStartY;
  if (Math.abs(dx) > SWIPE_THRESHOLD && Math.abs(dx) > Math.abs(dy)) {
    suppressClick = true; // swallow the trailing click so the swipe doesn't also open a link
    step(dx < 0 ? 1 : -1); // swipe left → next, swipe right → previous
  }
}

function onTrackPointerUp(e: PointerEvent): void {
  endDrag(e, true);
}

function onTrackPointerCancel(e: PointerEvent): void {
  endDrag(e, false);
}

/** Only acts after a completed swipe (see `suppressClick`); a genuine tap
 * reaches the anchor untouched and opens the image link as before. */
function onTrackClick(e: MouseEvent): void {
  if (!suppressClick) return;
  suppressClick = false;
  e.preventDefault();
}

let ro: ResizeObserver | null = null;
onMounted(() => {
  updateNav();
  if (track.value && 'ResizeObserver' in window) {
    ro = new ResizeObserver(() => updateNav());
    ro.observe(track.value);
  }
});
onBeforeUnmount(() => {
  ro?.disconnect();
  ro = null;
});
</script>

<template>
  <div v-if="resolved.length" class="rich-card image-card">
    <div class="image-card__header">
      <span class="image-card__query">{{ payload.query }}</span>
    </div>

    <div class="image-card__carousel">
      <button
        v-if="overflowing"
        type="button"
        class="image-card__nav image-card__nav--prev"
        :disabled="atStart"
        aria-label="Previous images"
        @click="step(-1)"
      >‹</button>

      <div
        ref="track"
        class="image-card__track"
        :class="{ 'image-card__track--dragging': isDragging }"
        @scroll="updateNav"
        @pointerdown="onTrackPointerDown"
        @pointermove="onTrackPointerMove"
        @pointerup="onTrackPointerUp"
        @pointercancel="onTrackPointerCancel"
        @click="onTrackClick"
      >
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
              draggable="false"
              @error="failed.add(idx)"
            />
            <div v-else class="image-card__img image-card__img--broken" aria-hidden="true">⊘</div>
          </component>
          <div v-if="img.verified" class="image-card__badge" role="img" aria-label="Verified">✓</div>
        </div>
      </div>

      <button
        v-if="overflowing"
        type="button"
        class="image-card__nav image-card__nav--next"
        :disabled="atEnd"
        aria-label="Next images"
        @click="step(1)"
      >›</button>
    </div>

    <div v-if="domains.length" class="image-card__sources-wrap">
      <button
        class="trace-pill"
        :class="{ 'trace-pill--open': sourcesOpen }"
        type="button"
        :aria-expanded="sourcesOpen"
        :aria-label="sourcesOpen ? 'Collapse sources' : 'Expand sources'"
        @click="sourcesOpen = !sourcesOpen"
      >
        <span class="trace-pill__dot" aria-hidden="true" />
        {{ domains.length }} source{{ domains.length === 1 ? '' : 's' }}
      </button>

      <div class="trace-body" :class="{ 'trace-body--open': sourcesOpen }">
        <div class="trace-body__inner">
          <div class="image-card__sources">
            <span v-for="domain in domains" :key="domain" class="image-card__src">
              <span
                class="image-card__src-ico"
                :style="{ background: pickColor(domain) }"
              >{{ initials(domain) }}</span>{{ domain }}</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped lang="scss">
/* Card-specific rules only; base .rich-card chrome lives globally in rich-card-base.scss. */

/* No card frame: the images ARE the content. Unset the base bg/border/padding and
   let the carousel span the full message column (mirrors weather-card). */
.rich-card.image-card {
  background: none;
  border: none;
  padding: 0;
  width: 100%;
  max-width: 100%;
}

.image-card__header {
  margin-bottom: 8px;
}

.image-card__query {
  font-size: 0.92rem;
  font-weight: 600;
  color: var(--text-primary);
  letter-spacing: -0.01em;
}

/* Carousel: a scroll-snap track (no visible scrollbar) paged by the edge buttons.
   1 slide per view on small screens, 2 on larger ones. */
.image-card__carousel {
  position: relative;
}

.image-card__track {
  display: flex;
  gap: 16px;
  overflow-x: auto;
  scroll-snap-type: x mandatory;
  scroll-behavior: smooth;
  scrollbar-width: none; /* Firefox */
  -ms-overflow-style: none; /* legacy Edge */
  /* Vertical page scroll stays a native browser gesture; horizontal drags are
     owned by the pointer-swipe handlers instead of native touch panning. */
  touch-action: pan-y;
  user-select: none;
  cursor: grab;

  &::-webkit-scrollbar {
    display: none; /* WebKit */
  }

  &--dragging {
    cursor: grabbing;
  }
}

.image-card__cell {
  position: relative;
  flex: 0 0 100%; /* 1-up on small screens */
  scroll-snap-align: start;
  transition: opacity 160ms ease;

  &:hover {
    opacity: 0.92;
  }
}

@media (min-width: 641px) {
  .image-card__cell {
    flex-basis: calc((100% - 16px) / 2); /* 2-up on larger screens */
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
  border-radius: 12px;
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
  top: 7px;
  right: 7px;
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
}

/* Edge navigation buttons — overlaid, vertically centred; hidden at their bound. */
.image-card__nav {
  position: absolute;
  top: 50%;
  transform: translateY(-50%);
  z-index: 2;
  width: 34px;
  height: 34px;
  border-radius: 50%;
  border: 1px solid var(--border);
  background: color-mix(in oklab, var(--bg-surface) 85%, transparent);
  backdrop-filter: blur(4px);
  color: var(--text-primary);
  font-size: 1.35rem;
  line-height: 1;
  padding: 0 0 3px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: opacity 160ms ease, background 160ms ease;

  &:hover {
    background: var(--bg-surface);
  }

  &:disabled {
    opacity: 0;
    pointer-events: none;
  }

  &--prev {
    left: 8px;
  }

  &--next {
    right: 8px;
  }
}

/* Sources fold: a "N sources" trace-pill (see global .trace-pill / .trace-body)
   that expands to the domain chips — mirrors the ACT trail's collapse. */
.image-card__sources-wrap {
  margin-top: 12px;
}

/* Deduped source pills — folds every thumbnail's origin into clean domain chips,
   mirroring the web-search card. Revealed inside the expandable trace-body. */
.image-card__sources {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
  font-family: var(--font-mono, 'JetBrains Mono', ui-monospace, monospace);
  font-size: 0.66rem;
  letter-spacing: 0.06em;
}

.image-card__src {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 3px 9px 3px 3px;
  border-radius: 999px;
  background: var(--bg-input);
  border: 1px solid var(--border);
  color: var(--text-secondary);
  unicode-bidi: isolate;
}

.image-card__src-ico {
  width: 16px;
  height: 16px;
  border-radius: 50%;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 0.52rem;
  font-weight: 600;
  color: white;
  flex-shrink: 0;
}
</style>
