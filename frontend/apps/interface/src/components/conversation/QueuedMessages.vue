<!-- Queued sends for one scope (the main spine or an open thread), rendered as
     faded text at the very tail — the last user turn, pushed down as the reply
     streams in. Each row clicks back into the composer for editing or drops out
     via its remove button. Dispatch is the session store's job. -->
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
 *  from the queue and hand the text to the matching InputDock. */
function edit(i: number): void {
  const text = items.value[i];
  queue.removeAt(props.threadId, i);
  void nextTick(() =>
    document.dispatchEvent(
      new CustomEvent('chalie:edit-queued', { detail: { turnId: props.threadId, text } }),
    ),
  );
}
</script>

<template>
  <div v-if="items.length" class="queued">
    <div
      v-for="(text, i) in items"
      :key="i"
      class="queued__row"
      :class="{ 'queued__row--lead': i === 0 }"
    >
      <button type="button" class="queued__msg" title="Click to edit" @click="edit(i)">
        <span class="user-text">{{ text }}</span>
      </button>
      <button
        type="button"
        class="queued__remove"
        aria-label="Remove queued message"
        @click="remove(i)"
      >
        <X :size="14" />
      </button>
    </div>
  </div>
</template>

<style scoped lang="scss">
.queued {
  display: flex;
  flex-direction: column;
}

.queued__row {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  width: 100%;
  max-width: var(--dock-width);
  margin-inline: auto;
  // Clear the avatar gutter + row gap so the faded text lines up under the
  // conversation's real user turns.
  padding-left: calc(var(--avatar-size) + 18px);
  margin-top: 6px;
  opacity: 0.7;
}

.queued__row--lead {
  margin-top: 30px;
}

.queued__msg {
  flex: 1;
  min-width: 0;
  text-align: left;
  padding: 0;
  border: none;
  background: none;
  cursor: pointer;
}

.queued__msg:hover .user-text {
  color: var(--text-primary);
}

.queued__remove {
  flex-shrink: 0;
  display: grid;
  place-items: center;
  width: 22px;
  height: 22px;
  padding: 0;
  border: none;
  border-radius: 6px;
  background: none;
  color: var(--text-tertiary);
  cursor: pointer;
  transition: color var(--duration-fast) ease, background var(--duration-fast) ease;
}

.queued__remove:hover {
  color: var(--error);
  background: var(--border);
}
</style>
