<!-- Queued sends for one scope (the main spine or an open thread), rendered as
     faded user rows at the very tail — the last user turn, pushed down as the
     reply streams in. Each row clicks back into the composer for editing or
     drops out via its remove button. Dispatch is the session store's job. -->
<script setup lang="ts">
import { computed, nextTick } from 'vue';
import { User, X } from '@lucide/vue';
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
      <div class="queued__gutter" aria-hidden="true">
        <span v-if="i === 0" class="queued__avatar"><User :size="15" /></span>
      </div>
      <div class="queued__body">
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
  </div>
</template>

<style scoped lang="scss">
// Faded (un-sent) user rows, aligned to the same avatar-gutter rhythm as
// TurnView so they read as the conversation's trailing user turn.
.queued {
  display: flex;
  flex-direction: column;
}

.queued__row {
  display: flex;
  gap: 18px;
  width: 100%;
  max-width: var(--dock-width);
  margin-inline: auto;
  margin-top: 6px;
  opacity: 0.7;
}

.queued__row--lead {
  margin-top: 30px;
}

.queued__gutter {
  width: var(--avatar-size);
  flex-shrink: 0;
  display: flex;
  justify-content: center;
  padding-top: 1px;
}

.queued__avatar {
  width: var(--avatar-size);
  height: var(--avatar-size);
  display: grid;
  place-items: center;
  border-radius: 50%;
  overflow: hidden;
  background: var(--bg-surface-2);
  border: 1px solid var(--border-strong);
  color: var(--text-secondary);
}

.queued__body {
  flex: 1;
  min-width: 0;
  display: flex;
  align-items: flex-start;
  gap: 8px;
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
