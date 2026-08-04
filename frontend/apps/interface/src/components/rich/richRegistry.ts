import type { Component } from 'vue';
import { defineAsyncComponent } from 'vue';

/**
 * Maps a tag base name (full slug stripped of a trailing `_<digits>` ordinal,
 * e.g. "image_preview" from "image_preview_2") to the card component that
 * renders it. news/search are text-only, so there is no ArticleCard. Cards are
 * async so each + its scoped styles code-split into their own chunk, keeping
 * the chat spine light until that card type appears.
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
  image_preview: {
    component: defineAsyncComponent(() => import('./ImagePreviewCard.vue')),
  },
};

/**
 * Resolve a card for a full tag (e.g. "image_preview_2" → ImagePreviewCard,
 * "weather_1" → WeatherCard) by stripping the trailing `_<digits>` ordinal.
 * Returns undefined for unknown bases so the caller falls back to synthesis
 * text.
 */
export function resolveRichCard(tag: string): RichCardEntry | undefined {
  const base = tag.replace(/_\d+$/, '');
  return richRegistry[base];
}
