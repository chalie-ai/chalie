import type { Component } from 'vue';

export interface RichCardEntry {
  component: Component;
}

// Populated by Task B0 (P1b). Empty in P1a → SegmentRenderer uses its text fallback.
export const richRegistry: Record<string, RichCardEntry> = {};

export function resolveRichCard(tag: string): RichCardEntry | undefined {
  return richRegistry[tag];
}
