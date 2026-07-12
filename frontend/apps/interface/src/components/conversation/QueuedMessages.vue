<!-- Pending (queued) sends for one scope, floating just above that scope's
     composer as blurred rounded chips — a clear "waiting to send" affordance.
     Each chip clicks back into the composer for editing; its x removes it.
     The chips live inside the InputDock, so the scope is the dock's turn_id and
     dispatch is the session store's job. -->
<script setup lang="ts">
import { computed, nextTick, reactive, watch } from 'vue';
import type { ComponentPublicInstance } from 'vue';
import { X } from '@lucide/vue';
import { useQueueStore } from '../../stores/queue';

const props = withDefaults(defineProps<{ threadId?: number | null }>(), { threadId: null });

const queue = useQueueStore();
const items = computed(() => queue.queuedFor(props.threadId));

// Per-item expand state and overflow detection for the 3-line clamp — both
// keyed by the same index used as the v-for key, so both reset whenever the
// queue's length changes (an item was queued, edited back out, or removed).
const expanded = reactive(new Set<number>());
const overflowing = reactive(new Set<number>());
const textRefs = new Map<number, HTMLElement>();

function setTextRef(i: number, el: Element | ComponentPublicInstance | null): void {
  if (el instanceof HTMLElement) {
    textRefs.set(i, el);
  } else {
    textRefs.delete(i);
  }
}

/** A clamped text node's scrollHeight still reports its full, unclamped
 *  height, so comparing it against clientHeight tells us whether the 3-line
 *  clamp is actually cutting the message off — the toggle only shows when it is. */
function measureOverflow(): void {
  overflowing.clear();
  for (const [i, el] of textRefs) {
    if (el.scrollHeight > el.clientHeight + 1) overflowing.add(i);
  }
}

watch(
  () => items.value.length,
  () => {
    expanded.clear();
    void nextTick(measureOverflow);
  },
  { immediate: true },
);

function remove(i: number): void {
  queue.removeAt(props.threadId, i);
}

/** Move a queued message back into its scope's composer for editing — drop it
 *  from the queue and hand its text to the matching InputDock. Any files it
 *  carried are dropped from the edit (re-attaching isn't supported here); the
 *  original behavior before D3b already never restored files to the strip. */
function edit(i: number): void {
  const text = items.value[i].text;
  queue.removeAt(props.threadId, i);
  void nextTick(() =>
    document.dispatchEvent(
      new CustomEvent('chalie:edit-queued', { detail: { turnId: props.threadId, text } }),
    ),
  );
}

function toggleExpanded(i: number, event: MouseEvent): void {
  event.stopPropagation();
  if (expanded.has(i)) expanded.delete(i);
  else expanded.add(i);
}
</script>

<template>
  <div v-if="items.length" class="pending">
    <div v-for="(entry, i) in items" :key="i" class="pending__row">
      <button
        type="button"
        class="pending__remove"
        aria-label="Remove queued message"
        @click="remove(i)"
      >
        <X :size="13" />
      </button>
      <div
        class="pending__chip"
        role="button"
        tabindex="0"
        title="Click to edit"
        @click="edit(i)"
        @keydown.enter="edit(i)"
        @keydown.space.prevent="edit(i)"
      >
        <span
          :ref="(el) => setTextRef(i, el)"
          class="pending__text"
          :class="{ 'pending__text--clamped': !expanded.has(i) }"
        >{{ entry.text }}</span>
        <button
          v-if="overflowing.has(i)"
          type="button"
          class="pending__toggle"
          :aria-expanded="expanded.has(i)"
          @click="toggleExpanded(i, $event)"
          @keydown="$event.stopPropagation()"
        >
          {{ expanded.has(i) ? 'Show less' : 'Read more' }}
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped lang="scss">
.pending {
  display: flex;
  flex-direction: column;
  gap: 6px;
  width: 100%;
  max-width: var(--dock-width);
  margin: 0 auto;
  padding: 0 4px 10px;
}

.pending__row {
  display: flex;
  align-items: center;
  gap: 8px;
}

.pending__remove {
  flex-shrink: 0;
  display: grid;
  place-items: center;
  width: 20px;
  height: 20px;
  padding: 0;
  border: none;
  border-radius: 50%;
  background: none;
  color: var(--text-tertiary);
  cursor: pointer;
  transition:
    color var(--duration-fast) ease,
    background var(--duration-fast) ease;
}

.pending__remove:hover {
  color: var(--error);
  background: var(--border);
}

// The floating chip: a translucent, blurred rounded box that hugs its text so a
// queued message reads as "pending" against the conversation behind it.
.pending__chip {
  min-width: 0;
  max-width: 100%;
  padding: 7px 14px;
  border: 1px solid var(--border);
  border-radius: 14px;
  background: color-mix(in oklab, var(--bg-surface-2) 62%, transparent);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  text-align: left;
  cursor: pointer;
  transition:
    border-color var(--duration-fast) ease,
    background var(--duration-fast) ease;
}

.pending__chip:hover {
  border-color: var(--border-strong);
  background: color-mix(in oklab, var(--bg-surface-2) 78%, transparent);
}

.pending__text {
  display: block;
  font-size: 0.875rem;
  line-height: 1.45;
  color: var(--text-secondary);
  white-space: pre-wrap;
  overflow-wrap: break-word;
}

.pending__text--clamped {
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.pending__chip:hover .pending__text {
  color: var(--text-primary);
}

.pending__toggle {
  display: inline-block;
  margin-top: 2px;
  padding: 0;
  border: none;
  background: none;
  font-size: 0.8125rem;
  font-weight: 500;
  color: var(--text-tertiary);
  cursor: pointer;
  transition: color var(--duration-fast) ease;
}

.pending__toggle:hover {
  color: var(--accent-primary);
}
</style>
