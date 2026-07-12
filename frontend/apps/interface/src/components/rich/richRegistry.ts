import type { Component } from 'vue';
import { defineAsyncComponent } from 'vue';

/**
 * Maps a tag PREFIX (before the first underscore, e.g. "weather" in "weather_1")
 * to the card component that renders it. news/search are text-only, so there is
 * no ArticleCard. Cards are async so each + its scoped styles code-split into
 * their own chunk, keeping the chat spine light until that card type appears.
 */
export interface RichCardEntry {
  component: Component;
}

const richRegistry: Record<string, RichCardEntry> = {
  weather: { component: defineAsyncComponent(() => import('./WeatherCard.vue')) },
  schedule: { component: defineAsyncComponent(() => import('./SchedulerCard.vue')) },
  list: { component: defineAsyncComponent(() => import('./ListCard.vue')) },
  timer: { component: defineAsyncComponent(() => import('./TimerCard.vue')) },
  calendar: { component: defineAsyncComponent(() => import('./CalendarCard.vue')) },
  contacts: { component: defineAsyncComponent(() => import('./ContactsCard.vue')) },
  image: { component: defineAsyncComponent(() => import('./ImageSearchCard.vue')) },
};

/**
 * Resolve a card for a full tag (e.g. "weather_1" → WeatherCard) by its prefix.
 * Returns undefined for unknown prefixes so the caller falls back to synthesis text.
 */
export function resolveRichCard(tag: string): RichCardEntry | undefined {
  return richRegistry[tag.split('_')[0] ?? ''];
}
