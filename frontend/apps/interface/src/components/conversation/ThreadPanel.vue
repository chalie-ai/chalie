<!-- Slide-over thread panel: a focused, full-height view of one thread. Replaces
     the old inline accordion — a pill or the Reply action opens it via
     session.panelThreadId. Renders the thread's rows with TurnView and carries
     its own reply dock (turn_id pinned), so replies append to this thread. -->
<script setup lang="ts">
import { ref, computed, watch, nextTick, onMounted, onBeforeUnmount } from 'vue';
import { ArrowLeft } from '@lucide/vue';
import { useSessionStore } from '../../stores/session';
import { useConversationStore } from '../../stores/conversation';
import type { ConversationForm } from '../../stores/conversation';
import TurnView from './TurnView.vue';
import InputDock from '../layout/InputDock.vue';

const session = useSessionStore();
const convo = useConversationStore();

const open = computed(() => session.panelThreadId != null);

const thread = computed(() =>
  session.panelThreadId == null
    ? null
    : (convo.threads.find((t) => t.turn_id === session.panelThreadId) ?? null),
);

// Rows of the open thread. A reply tags its live forms with the thread's turn_id
// up-front (session._startTurn), so this stays current mid-stream.
const panelForms = computed<ConversationForm[]>(() =>
  session.panelThreadId == null
    ? []
    : convo.forms.filter((f) => f.turnId === session.panelThreadId),
);

const heading = computed(() => thread.value?.gist || thread.value?.preview || 'Thread');
const showLoader = computed(() => session.threadExpanding && !panelForms.value.length);

const bodyRef = ref<HTMLElement | null>(null);

function close(): void {
  session.closeThreadPanel();
}

// Follow live reply growth: pin the body to its bottom whenever the open thread's
// forms change. flush:'post' lets it measure settled height.
watch(
  panelForms,
  () => {
    if (!open.value) return;
    nextTick(() => {
      const el = bodyRef.value;
      if (el) el.scrollTop = el.scrollHeight;
    });
  },
  { deep: true, flush: 'post' },
);

function onKeydown(e: KeyboardEvent): void {
  if (e.key === 'Escape' && open.value) close();
}

// Lock the page behind the panel so a background scroll can't bleed through the
// dimmed strip.
watch(open, (isOpen) => {
  document.body.style.overflow = isOpen ? 'hidden' : '';
});

onMounted(() => document.addEventListener('keydown', onKeydown));
onBeforeUnmount(() => {
  document.removeEventListener('keydown', onKeydown);
  document.body.style.overflow = '';
});
</script>

<template>
  <Transition name="thread-panel">
    <div v-if="open" class="thread-panel-overlay" @click.self="close">
      <aside class="thread-panel" role="dialog" aria-modal="true" :aria-label="heading">
        <header class="thread-panel__header">
          <button
            class="thread-panel__back"
            type="button"
            aria-label="Back to conversation"
            @click="close"
          >
            <ArrowLeft :size="18" />
          </button>
          <span class="thread-panel__crumb-label">Thread</span>
          <span class="thread-panel__crumb-title">{{ heading }}</span>
        </header>

        <div ref="bodyRef" class="thread-panel__body">
          <div v-if="showLoader" class="thread-panel__loader">
            <output class="thread-panel__spinner" aria-label="Loading thread" />
          </div>
          <TurnView v-else :forms="panelForms" :can-reply="false" />
        </div>

        <InputDock v-if="session.panelThreadId != null" :turn-id="session.panelThreadId" />
      </aside>
    </div>
  </Transition>
</template>

<style scoped lang="scss">
.thread-panel-overlay {
  position: fixed;
  inset: 0;
  z-index: 150;
  display: flex;
  justify-content: flex-end;
  // Dimmed sliver of the main conversation shows through on the left.
  background: color-mix(in oklab, var(--bg) 55%, transparent);
  backdrop-filter: blur(6px) saturate(120%);
  -webkit-backdrop-filter: blur(6px) saturate(120%);
}

.thread-panel {
  width: 95%;
  height: 100%;
  display: flex;
  flex-direction: column;
  background: var(--bg);
  border-left: 1px solid var(--border-strong);
  box-shadow: -24px 0 60px color-mix(in oklab, #000 40%, transparent);
}

.thread-panel__header {
  flex-shrink: 0;
  display: flex;
  align-items: center;
  gap: var(--space-sm);
  padding: var(--space-md) var(--space-lg);
  border-bottom: 1px solid var(--border);
}

.thread-panel__back {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 30px;
  height: 30px;
  border: none;
  border-radius: 8px;
  background: none;
  color: var(--text-secondary);
  cursor: pointer;
  transition: color var(--duration-fast), background var(--duration-fast);
}

.thread-panel__back:hover {
  color: var(--text-primary);
  background: color-mix(in oklab, var(--text-primary) 7%, transparent);
}

.thread-panel__crumb-label {
  flex-shrink: 0;
  font-size: 13px;
  font-weight: 600;
  color: var(--violet);
}

.thread-panel__crumb-title {
  min-width: 0;
  font-size: 13px;
  color: var(--text-tertiary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.thread-panel__body {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  padding: var(--space-md) 0 var(--space-lg);
}

.thread-panel__loader {
  display: flex;
  justify-content: center;
  padding: 32px 0;
}

.thread-panel__spinner {
  width: 20px;
  height: 20px;
  border: 2px solid color-mix(in oklab, var(--violet) 20%, transparent);
  border-top-color: var(--violet);
  border-radius: 50%;
  animation: thread-panel-spin 0.7s linear infinite;
}

@keyframes thread-panel-spin {
  to {
    transform: rotate(360deg);
  }
}

// Slide-in: the overlay scrim fades, the panel slides from the right.
.thread-panel-enter-active,
.thread-panel-leave-active {
  transition: opacity 200ms ease;
}

.thread-panel-enter-active .thread-panel,
.thread-panel-leave-active .thread-panel {
  transition: transform 280ms cubic-bezier(0.22, 1, 0.36, 1);
}

.thread-panel-enter-from,
.thread-panel-leave-to {
  opacity: 0;
}

.thread-panel-enter-from .thread-panel,
.thread-panel-leave-to .thread-panel {
  transform: translateX(100%);
}

@media (prefers-reduced-motion: reduce) {
  .thread-panel-enter-active .thread-panel,
  .thread-panel-leave-active .thread-panel {
    transition: none;
  }
}
</style>
