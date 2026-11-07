import type { Component } from 'vue';
import { defineAsyncComponent } from 'vue';
import type { ConversationSegment } from '../../api/conversation';

/**
 * Maps a tag base name (full slug stripped of a trailing `_<digits>` ordinal,
 * e.g. "image_preview" from "image_preview_2") to the card component that
 * renders it. news/search are text-only, so there is no ArticleCard. Cards stay
 * code-split, each + its scoped styles in their own chunk, keeping the chat
 * spine light until that card type appears. A history page preloads its cards
 * (preloadRichCards) so they mount with their turn instead of a frame later.
 */
export interface RichCardEntry {
  component: Component;
}

const loaders: Record<string, () => Promise<{ default: Component }>> = {
  weather: () => import('./WeatherCard.vue'),
  schedule: () => import('./SchedulerCard.vue'),
  list: () => import('./ListCard.vue'),
  timer: () => import('./TimerCard.vue'),
  calendar: () => import('./CalendarCard.vue'),
  contacts: () => import('./ContactsCard.vue'),
  image_preview: () => import('./ImagePreviewCard.vue'),
};

/** Fallback for card types not loaded yet: renders once its chunk arrives. */
const asyncCards: Record<string, Component> = Object.fromEntries(
  Object.entries(loaders).map(([base, load]) => [base, defineAsyncComponent(load)]),
);

const loadedCards = new Map<string, Component>();

function baseOf(tag: string): string {
  return tag.replace(/_\d+$/, '');
}

/**
 * Load the chunks for every rich card in `segments` that isn't loaded yet, so
 * a render after this resolves mounts those cards synchronously. A failed load
 * is left to the async fallback, which retries and reports its own error.
 */
export async function preloadRichCards(segments: ConversationSegment[]): Promise<void> {
  const bases = new Set(
    segments
      .filter((seg) => seg.type === 'rich')
      .map((seg) => baseOf(seg.tag ?? ''))
      .filter((base) => base in loaders && !loadedCards.has(base)),
  );
  await Promise.allSettled(
    [...bases].map(async (base) => {
      loadedCards.set(base, (await loaders[base]()).default);
    }),
  );
}

/**
 * Resolve a card for a full tag (e.g. "image_preview_2" → ImagePreviewCard,
 * "weather_1" → WeatherCard) by stripping the trailing `_<digits>` ordinal.
 * Returns undefined for unknown bases so the caller falls back to synthesis
 * text.
 */
export function resolveRichCard(tag: string): RichCardEntry | undefined {
  const base = baseOf(tag);
  const component = loadedCards.get(base) ?? asyncCards[base];
  return component ? { component } : undefined;
}
