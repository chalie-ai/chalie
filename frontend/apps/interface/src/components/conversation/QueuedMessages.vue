<!-- Pending (queued) sends for one scope, floating just above that scope's
     composer as blurred rounded chips — a clear "waiting to send" affordance.
     Each chip clicks back into the composer for editing; its x removes it.
     The chips live inside the InputDock, so the scope is the dock's turn_id and
     dispatch is the session store's job. -->
<script setup lang="ts">
import { computed, nextTick } from 'vue';
import { X } from '@lucide/vue';
import { useQueueStore } from '../../stores/queue';

const props = withDefaults(defineProps<{ threadId?: number | null }>(), { threadId: null });

const queue = useQueueStore();
const items = computed(() => queue.queuedFor(props.threadId));

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
      <button type="button" class="pending__chip" title="Click to edit" @click="edit(i)">
        <span class="pending__text">{{ entry.text }}</span>
      </button>
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

.pending__chip:hover .pending__text {
  color: var(--text-primary);
}
</style>
