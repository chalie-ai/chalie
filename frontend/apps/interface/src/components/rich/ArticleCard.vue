<!--
  ArticleCard — shared rich-media card for News + Web Search results.

  Serves both the `news` and `search` tag prefixes. Layout: optional image
  thumbnail, a synthesis paragraph (rendered as chalie-markup), and a row of
  deduplicated domain pills (max 4 shown, remainder as "+ N more").

  The .rich-card base chrome (background, border, padding, animation) is
  declared globally in base_card.css — do not redeclare those rules here.
-->
<script setup lang="ts">
import { ref, computed } from 'vue';
import { renderMarkup } from '../../composables/useMarkup';

// ── Payload contract ───────────────────────────────────────────────────────

export interface ArticleResult {
  /** Source URL — used to derive the domain pill label. */
  url?: string;
  /** Fallback source label when url is absent. */
  source?: string;
  /** Article headline. */
  title?: string;
  /** Short excerpt / snippet. */
  snippet?: string;
  /** Per-article thumbnail (unused in current layout but typed for completeness). */
  image_url?: string;
  /** ISO-8601 publish date string. */
  published_at?: string;
}

export interface ArticlePayload {
  /** Array of article results. */
  results: ArticleResult[];
  /** Optional hero image shown alongside the synthesis. */
  image_url?: string;
}

// ── Props ──────────────────────────────────────────────────────────────────

const props = defineProps<{
  payload: ArticlePayload;
  synthesis?: string;
}>();

// ── Color palette (exact order from article.js) ────────────────────────────

const SOURCE_COLORS: readonly string[] = [
  '#4ea2ff', '#FF2FD1', '#F2C94C', '#00F0FF', '#34d399',
  '#fb7185', '#8A5CFF', '#fbbf24', '#6E3FE6', '#B07CFF',
] as const;

/**
 * Hash-deterministic color pick.
 *
 * Exact port of pickColor() from article.js:
 *   hash = Math.trunc((hash << 5) - hash + codePointAt(i))
 *   color = SOURCE_COLORS[Math.abs(hash) % SOURCE_COLORS.length]
 */
function pickColor(name: string): string {
  let hash = 0;
  for (let i = 0; i < name.length; i++) {
    hash = Math.trunc((hash << 5) - hash + (name.codePointAt(i) ?? 0));
  }
  return SOURCE_COLORS[Math.abs(hash) % SOURCE_COLORS.length] as string;
}

/**
 * Two-character initials for the pill icon.
 * Exact port of initials() from article.js.
 */
function initials(name: string): string {
  return name
    .replace(/^https?:\/\/(www\.)?/, '')
    .replace(/\.com|\.org|\.net|\.io|\.eu|\.co\.uk/g, '')
    .slice(0, 2)
    .toUpperCase();
}

/**
 * Extract hostname from a URL, stripping leading www.
 * Exact port of extractDomain() from article.js.
 */
function extractDomain(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, '');
  } catch {
    return url;
  }
}

// ── Image visibility (onerror equivalent) ─────────────────────────────────

/** Set to true when the hero image fails to load — hides the <img> element. */
const imageErrored = ref<boolean>(false);

function onImageError(): void {
  // Mirrors legacy: img.onerror = () => img.remove()
  imageErrored.value = true;
}

// ── Domain pills ───────────────────────────────────────────────────────────

/** Unique domains derived from results, preserving insertion order. */
const domains = computed<string[]>(() => {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const r of props.payload.results ?? []) {
    const d: string = r.url ? extractDomain(r.url) : (r.source ?? '');
    if (!d || seen.has(d)) continue;
    seen.add(d);
    out.push(d);
  }
  return out;
});

/** First 4 domains shown as pills — mirrors legacy: domains.slice(0, 4). */
const shownDomains = computed<string[]>(() => domains.value.slice(0, 4));

/** Count of domains not shown — mirrors legacy: domains.length - shown.length. */
const remainingCount = computed<number>(() => domains.value.length - shownDomains.value.length);

// ── Synthesis markup ───────────────────────────────────────────────────────

/** Rendered HTML for the synthesis paragraph (via the shared markup pipeline). */
const synthesisHtml = computed<string>(() => renderMarkup(props.synthesis ?? ''));

// ── Hero image ─────────────────────────────────────────────────────────────

const imageUrl = computed<string | null>(() => props.payload.image_url ?? null);
</script>

<template>
  <div class="rich-card article-card">
    <div class="article-card__body">
      <!-- Synthesis paragraph — renderMarkup mirrors renderMarkupTo from article.js -->
      <p
        class="article-card__dek chalie-markup"
        v-html="synthesisHtml"
      />

      <!-- Hero image — shown only when present and not errored (mirrors img.onerror = () => img.remove()) -->
      <img
        v-if="imageUrl && !imageErrored"
        class="article-card__img"
        :src="imageUrl"
        alt=""
        loading="lazy"
        @error="onImageError"
      />

      <!-- Domain pills — max 4 shown + optional "+ N more" overflow label -->
      <div v-if="shownDomains.length > 0" class="article-card__sources">
        <span
          v-for="domain in shownDomains"
          :key="domain"
          class="article-card__src"
        >
          <span
            class="article-card__src-ico"
            :style="{ background: pickColor(domain) }"
          >{{ initials(domain) }}</span>{{ domain }}</span>

        <span v-if="remainingCount > 0" class="article-card__meta">
          + {{ remainingCount }} more
        </span>
      </div>
    </div>
  </div>
</template>

<style scoped lang="scss">
/*
 * Article-card-specific rules only.
 * .rich-card base chrome (bg, border, padding, animation) lives in base_card.css (global).
 * All color values use CSS custom properties — works in both light and dark themes (Rule 7).
 */

.article-card {
  background: none;
  border: none;
  padding: 0;
  width: 100%;
  max-width: 100%;
}

.article-card__img {
  display: block;
  max-width: min(500px, 90%);
  max-height: 500px;
  width: auto;
  height: auto;
  border-radius: 10px;
}

.article-card__body {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.article-card__dek {
  font-size: 1rem;
  color: var(--text-primary);
  margin: 0;
  line-height: 1.55;
  letter-spacing: -0.005em;

  // Synthesis is injected via v-html, so its <em> nodes carry no scope
  // attribute — :deep() is required for the scoped rule to reach them
  // (legacy article.css:37-40 was global and matched without it).
  :deep(em) {
    color: var(--text-secondary);
    font-style: italic;
  }
}

.article-card__sources {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
  font-family: var(--font-mono, 'JetBrains Mono', ui-monospace, monospace);
  font-size: 0.66rem;
  color: var(--text-tertiary);
  letter-spacing: 0.06em;
}

.article-card__src {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 3px 9px 3px 3px;
  border-radius: 999px;
  background: var(--bg-input);
  border: 1px solid var(--border);
  color: var(--text-secondary);
}

.article-card__src-ico {
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

.article-card__meta {
  margin-left: 4px;
  color: var(--text-tertiary);
}
</style>
