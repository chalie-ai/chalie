import type { Component } from 'vue';
import { defineAsyncComponent } from 'vue';

/**
 * Rich-media card registry (Task B0, P1b).
 *
 * Port of legacy frontend/interface/rich_media/registry.js: maps a tag PREFIX
 * (the part before the first underscore, e.g. "weather" in "weather_1") to the
 * card component that renders it. Eight prefixes → seven components (news and
 * search share ArticleCard).
 *
 * Cards are loaded via `defineAsyncComponent` so each card + its scoped styles
 * code-split into their own chunk — the chat spine stays light until a card of
 * that type actually appears.
 */
export interface RichCardEntry {
  component: Component;
}

const richRegistry: Record<string, RichCardEntry> = {
  weather: { component: defineAsyncComponent(() => import('./WeatherCard.vue')) },
  news: { component: defineAsyncComponent(() => import('./ArticleCard.vue')) },
  search: { component: defineAsyncComponent(() => import('./ArticleCard.vue')) },
  schedule: { component: defineAsyncComponent(() => import('./SchedulerCard.vue')) },
  list: { component: defineAsyncComponent(() => import('./ListCard.vue')) },
  timer: { component: defineAsyncComponent(() => import('./TimerCard.vue')) },
  calendar: { component: defineAsyncComponent(() => import('./CalendarCard.vue')) },
  contacts: { component: defineAsyncComponent(() => import('./ContactsCard.vue')) },
};

/**
 * Resolve a rich card for a full tag string (e.g. "weather_1" → WeatherCard).
 * Splits on the first underscore to get the prefix, exactly like the legacy
 * `renderCard` did (`tag.split('_')[0]`). Returns undefined for unknown
 * prefixes so SegmentRenderer falls back to rendering the synthesis text.
 */
export function resolveRichCard(tag: string): RichCardEntry | undefined {
  const prefix = tag.split('_')[0] ?? '';
  return richRegistry[prefix];
}
