<script setup lang="ts">
/**
 * QuoteDialog — a quote block in the composer, opened to read in full: the
 * quoted text, then the reply to it. Read-only; a block is never edited, only
 * removed from its row. Closes on the ×, a click outside, or Escape — which
 * goes no further, so it never closes the thread panel behind it.
 */
import { computed, nextTick, onBeforeUnmount, onMounted, ref } from 'vue';
import { X } from '@lucide/vue';
import type { QuoteBlock } from '../../utils/quoteBlocks';

const props = defineProps<{ block: QuoteBlock }>();
const emit = defineEmits<{ close: [] }>();

const quoted = computed(() =>
  props.block.quote
    .split('\n')
    .map((line) => `> ${line}`)
    .join('\n'),
);

const closeRef = ref<HTMLButtonElement | null>(null);
// Focus goes back to whatever opened the dialog — the row it was read from.
const opener = document.activeElement instanceof HTMLElement ? document.activeElement : null;

function onKeydown(e: KeyboardEvent): void {
  if (e.key !== 'Escape') return;
  e.stopPropagation();
  emit('close');
}

onMounted(() => void nextTick(() => closeRef.value?.focus()));
onBeforeUnmount(() => opener?.focus());
</script>

<template>
  <Teleport to="body">
    <div class="quote-dialog" @click.self="emit('close')">
      <div
        class="quote-dialog__card"
        role="dialog"
        aria-modal="true"
        aria-labelledby="quoteDialogTitle"
        tabindex="-1"
        @keydown="onKeydown"
      >
        <header class="quote-dialog__head">
          <h2 id="quoteDialogTitle" class="quote-dialog__title">Reply</h2>
          <button
            ref="closeRef"
            class="quote-dialog__close"
            type="button"
            aria-label="Close"
            @click="emit('close')"
          >
            <X :size="16" />
          </button>
        </header>
        <div class="quote-dialog__body">
          <p class="quote-dialog__quote">{{ quoted }}</p>
          <p v-if="block.reply" class="quote-dialog__reply">{{ block.reply }}</p>
        </div>
      </div>
    </div>
  </Teleport>
</template>

<style scoped lang="scss">
.quote-dialog {
  position: fixed;
  inset: 0;
  z-index: 1200;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: var(--space-lg);
  background: color-mix(in srgb, var(--bg) 60%, transparent);
  animation: fade var(--dur-2) var(--ease-out);
}

.quote-dialog__card {
  display: flex;
  flex-direction: column;
  gap: var(--space-md);
  width: min(520px, 100%);
  max-height: 80vh;
  padding: var(--space-md) var(--space-md) var(--space-lg);
  border-radius: 8px;
  background: var(--surface);
  outline: none;
  animation: rise var(--dur-2) var(--ease-out);
}

.quote-dialog__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-sm);
}

.quote-dialog__title {
  margin: 0;
  font-size: var(--fs-title);
  font-weight: 600;
  line-height: 1.2;
}

.quote-dialog__close {
  display: grid;
  place-items: center;
  flex-shrink: 0;
  width: var(--control-h);
  height: var(--control-h);
  padding: 0;
  background: var(--surface-2);
  color: var(--text);
  cursor: pointer;
  transition: background-color var(--dur-1) var(--ease-out);

  &:hover {
    background: var(--cell);
  }
}

// The quote and its reply under one rule down the left, drawn as a fill.
.quote-dialog__body {
  display: flex;
  flex-direction: column;
  gap: var(--space-md);
  min-height: 0;
  overflow-y: auto;
  padding-left: 12px;
  background: linear-gradient(var(--line), var(--line)) left / 2px 100% no-repeat;
}

.quote-dialog__quote,
.quote-dialog__reply {
  margin: 0;
  white-space: pre-wrap;
  overflow-wrap: break-word;
  line-height: 1.55;
}

.quote-dialog__quote {
  color: var(--muted);
}
</style>
