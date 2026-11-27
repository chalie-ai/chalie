<!-- The quote blocks waiting in a composer, one row each above the message
     field: the quote on one line, a tag when it carries a reply, and an × to
     remove it before the message is sent. A row opens to read in full; it is
     never edited. -->
<script setup lang="ts">
import { computed, ref } from 'vue';
import { Quote, X } from '@lucide/vue';
import type { QuoteBlock } from '../../utils/quoteBlocks';
import QuoteDialog from './QuoteDialog.vue';

const props = defineProps<{ blocks: readonly QuoteBlock[] }>();
const emit = defineEmits<{ remove: [index: number] }>();

const openIndex = ref<number | null>(null);
const openBlock = computed(() =>
  openIndex.value == null ? null : (props.blocks[openIndex.value] ?? null),
);
</script>

<template>
  <ul v-if="blocks.length" class="quote-rows" aria-label="Quotes">
    <li v-for="(block, i) in blocks" :key="i" class="quote-rows__row">
      <button type="button" class="quote-rows__open" title="Open quote" @click="openIndex = i">
        <Quote class="quote-rows__icon" :size="16" aria-hidden="true" />
        <span class="quote-rows__quote">“{{ block.quote }}”</span>
        <span v-if="block.reply" class="tag quote-rows__tag">With reply</span>
      </button>
      <button
        type="button"
        class="quote-rows__remove"
        aria-label="Remove quote"
        @click="emit('remove', i)"
      >
        <X :size="16" />
      </button>
    </li>
  </ul>
  <QuoteDialog v-if="openBlock" :block="openBlock" @close="openIndex = null" />
</template>

<style scoped lang="scss">
.quote-rows {
  display: flex;
  flex-direction: column;
  gap: var(--space-xs);
  width: 100%;
  max-width: var(--dock-width);
  margin: 0 auto;
  padding: 0 4px var(--space-sm);
  list-style: none;
}

.quote-rows__row {
  display: flex;
  align-items: center;
  min-width: 0;
  background: var(--surface-2);
  transition: background var(--dur-1) var(--ease-out);

  &:hover {
    background: var(--cell);
  }
}

.quote-rows__open {
  display: flex;
  flex: 1;
  align-items: center;
  gap: var(--space-sm);
  min-width: 0;
  min-height: var(--control-h);
  padding: 0 var(--space-xs) 0 12px;
  background: none;
  color: var(--muted);
  text-align: left;
  cursor: pointer;
  transition: color var(--dur-1) var(--ease-out);

  &:hover {
    color: var(--text);
  }
}

.quote-rows__icon {
  flex-shrink: 0;
}

// One line, cut with an ellipsis; a quote's own line breaks fold into it.
.quote-rows__quote {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  font-size: var(--fs-body);
  white-space: nowrap;
  text-overflow: ellipsis;
}

.quote-rows__tag {
  flex-shrink: 0;
  background: var(--cell);
}

.quote-rows__row:hover .quote-rows__tag {
  background: var(--surface-2);
}

.quote-rows__remove {
  display: grid;
  place-items: center;
  flex-shrink: 0;
  width: var(--control-h);
  height: var(--control-h);
  padding: 0;
  background: none;
  color: var(--muted);
  cursor: pointer;
  transition: color var(--dur-1) var(--ease-out);

  &:hover {
    color: var(--text);
  }
}
</style>
