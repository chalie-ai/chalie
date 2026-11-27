<!-- The quote blocks a sent or queued message opens with, read-only: each on
     its own fill, the quoted text in bold over the reply to it. Compact cuts
     each to two lines, for a queued message's preview. -->
<script setup lang="ts">
import type { QuoteBlock } from '../../utils/quoteBlocks';

defineProps<{ blocks: readonly QuoteBlock[]; compact?: boolean }>();
</script>

<template>
  <ul class="quote-blocks" :class="{ 'quote-blocks--compact': compact }">
    <li v-for="(block, i) in blocks" :key="i" class="quote-blocks__item">
      <span class="quote-blocks__quote">{{ block.quote }}</span>
      <span v-if="block.reply" class="quote-blocks__reply">{{ block.reply }}</span>
    </li>
  </ul>
</template>

<style scoped lang="scss">
.quote-blocks {
  display: flex;
  flex-direction: column;
  gap: var(--space-xs);
  margin: 0;
  padding: 0;
  list-style: none;
}

// A shade darker than the bubble it sits in: the page's own surface on the
// dark theme, the cell tone on the light one.
.quote-blocks__item {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: var(--space-sm) 12px;
  border-radius: 6px;
  background: var(--surface);
  white-space: pre-wrap;
  overflow-wrap: break-word;
  line-height: 1.55;
}

[data-theme='light'] .quote-blocks__item {
  background: var(--cell);
}

.quote-blocks__quote {
  font-weight: 600;
}

.quote-blocks__reply {
  font-weight: 500;
}

.quote-blocks--compact .quote-blocks__quote,
.quote-blocks--compact .quote-blocks__reply {
  display: -webkit-box;
  -webkit-line-clamp: 2;
  line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
</style>
