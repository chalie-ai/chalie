<!-- The running-subtasks ring: a turn that started subtasks still running shows
     a round button at its bottom right, just above the footer icons. Clicking
     it lists those subtasks; picking one opens its transcript in the thread
     panel, the same way a delegate pill does. -->
<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from 'vue';
import type { DelegateRef } from '@chalie/shared';
import { useSessionStore } from '../../stores/session';

/** The running subtasks — non-empty: the parent unmounts this component when
 *  none is left. `summary` is the call's one-line label, empty when the call
 *  had none. */
defineProps<{ subtasks: { delegate: DelegateRef; summary: string }[] }>();

const open = ref(false);
const wrapRef = ref<HTMLDivElement | null>(null);

/** Open the picked subtask's transcript in the thread panel, then close. */
function pick(delegate: DelegateRef): void {
  useSessionStore().openDelegatePanel(delegate);
  open.value = false;
}

/** Close on a click outside the ring (same pattern as InputDock's thinking
 *  menu) — the trigger's own click is stopped, so it can't re-close. */
function onDocumentClick(e: MouseEvent): void {
  const target = e.target as Node;
  if (open.value && !wrapRef.value?.contains(target)) open.value = false;
}

function onKeydown(e: KeyboardEvent): void {
  if (e.key === 'Escape' && open.value) open.value = false;
}

onMounted(() => {
  document.addEventListener('click', onDocumentClick);
  document.addEventListener('keydown', onKeydown);
});

onBeforeUnmount(() => {
  document.removeEventListener('click', onDocumentClick);
  document.removeEventListener('keydown', onKeydown);
});
</script>

<template>
  <div ref="wrapRef" class="subtask-ring">
    <button
      type="button"
      class="subtask-ring__trigger"
      aria-haspopup="menu"
      :aria-expanded="open"
      :aria-label="`${subtasks.length} running subtask(s)`"
      title="Running subtasks"
      @click.stop="open = !open"
    >
      <svg class="subtask-ring__ring" viewBox="0 0 32 32" aria-hidden="true">
        <circle cx="16" cy="16" r="13" fill="none" stroke="var(--violet)" stroke-width="2.5" />
      </svg>
      <span class="subtask-ring__count">{{ subtasks.length }}</span>
    </button>

    <div v-if="open" class="subtask-ring__menu" role="menu">
      <button
        v-for="s in subtasks"
        :key="s.delegate.turn_id"
        type="button"
        role="menuitem"
        class="subtask-ring__item"
        @click="pick(s.delegate)"
      >
        <span class="subtask-ring__name">Subtask #{{ s.delegate.turn_id }}</span>
        <span v-if="s.summary" class="subtask-ring__summary">{{ s.summary }}</span>
      </button>
    </div>
  </div>
</template>

<style scoped lang="scss">
.subtask-ring {
  position: relative;
}

.subtask-ring__trigger {
  width: 32px;
  height: 32px;
  border: 0;
  border-radius: 50%;
  background: var(--surface-2);
  display: grid;
  place-items: center;
  padding: 0;
  cursor: pointer;
  color: var(--text);
  font-family: var(--font-mono);
  font-size: var(--fs-mono);
  font-weight: 600;
  transition: background-color var(--dur-1) var(--ease-out);
}

.subtask-ring__trigger:hover,
.subtask-ring__trigger:focus-visible {
  background: var(--line);
}

.subtask-ring__trigger:focus-visible {
  outline: 1.5px solid var(--muted);
}

/* The ring and its count share the trigger's single grid cell, centred. */
.subtask-ring__ring,
.subtask-ring__count {
  grid-area: 1 / 1;
}

.subtask-ring__ring {
  width: 32px;
  height: 32px;
}

.subtask-ring__menu {
  position: absolute;
  bottom: 100%;
  right: 0;
  margin-bottom: 6px;
  min-width: 200px;
  max-width: 320px;
  z-index: 200;
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: var(--space-xs);
  background: var(--surface);
}

.subtask-ring__item {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 2px;
  text-align: left;
  background: transparent;
  border: 0;
  cursor: pointer;
  padding: var(--space-sm) var(--space-md);
  font-size: var(--fs-body);
  color: var(--text);
}

.subtask-ring__item:hover {
  background: var(--surface-2);
}

.subtask-ring__name {
  font-weight: 600;
}

.subtask-ring__summary {
  font-size: var(--fs-mono);
  color: var(--muted);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 100%;
}
</style>
