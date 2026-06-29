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
import QueuedMessages from './QueuedMessages.vue';
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

onMounted(() => document.addEventListener('keydown', onKeydown));
onBeforeUnmount(() => document.removeEventListener('keydown', onKeydown));
</script>

<template>
  <Transition name="thread-panel">
    <aside
      v-if="open"
      class="thread-panel"
      role="dialog"
      aria-modal="true"
      :aria-label="heading"
    >
      <header class="thread-panel__header">
        <button
          class="thread-panel__back"
          type="button"
          aria-label="Back to conversation"
          @click="close"
        >
          <ArrowLeft :size="16" />
          <span>Chalie</span>
        </button>
        <div class="thread-panel__divider" aria-hidden="true" />
        <svg
          class="thread-panel__fork-glyph"
          width="14"
          height="14"
          viewBox="0 0 24 24"
          fill="none"
          stroke="var(--violet-light)"
          stroke-width="2.2"
          stroke-linecap="round"
          stroke-linejoin="round"
          aria-hidden="true"
        >
          <circle cx="6" cy="6" r="3" />
          <circle cx="18" cy="18" r="3" />
          <path d="M6 9c0 6 6 3 6 9" />
        </svg>
        <span class="thread-panel__title">Thread: {{ heading }}</span>
      </header>

      <div ref="bodyRef" class="thread-panel__body">
        <div v-if="showLoader" class="thread-panel__loader">
          <output class="thread-panel__spinner" aria-label="Loading thread" />
        </div>
        <template v-else>
          <TurnView :forms="panelForms" :can-reply="false" />
          <!-- Thread-scoped queued sends — faded trailing turn, drains into this thread. -->
          <QueuedMessages :thread-id="session.panelThreadId" />
        </template>
      </div>

      <InputDock v-if="session.panelThreadId != null" :turn-id="session.panelThreadId" />
    </aside>
  </Transition>
</template>

<style scoped lang="scss">
.thread-panel {
  position: fixed;
  top: 56px;
  right: 0;
  bottom: 0;
  width: 95%;
  // Above the base feed + footer dock (both z-index:100, dimmed behind the
  // panel) but below true modals (search overlay, permission cards).
  z-index: 120;
  display: flex;
  flex-direction: column;
  background: var(--bg);
  border-left: 1px solid var(--border);
  border-radius: 16px 0 0 16px;
  box-shadow: -26px 0 64px rgba(0, 0, 0, 0.6);
  overflow: hidden;
}

.thread-panel__header {
  flex-shrink: 0;
  display: flex;
  align-items: center;
  gap: 11px;
  height: 46px;
  padding: 0 26px;
  border-bottom: 1px solid var(--border);
}

.thread-panel__back {
  display: flex;
  align-items: center;
  gap: 7px;
  margin-left: -4px;
  padding: 5px 9px 5px 6px;
  border: none;
  border-radius: 8px;
  background: none;
  color: var(--text-tertiary);
  font: 500 13px Inter, system-ui, sans-serif;
  cursor: pointer;
  transition: color var(--duration-fast), background var(--duration-fast);
}

.thread-panel__back:hover {
  color: var(--text-primary);
  background: var(--border);
}

.thread-panel__divider {
  width: 1px;
  height: 16px;
  background: var(--border-strong);
  flex-shrink: 0;
}

.thread-panel__fork-glyph {
  flex-shrink: 0;
}

.thread-panel__title {
  font: 600 14px Inter, system-ui, sans-serif;
  letter-spacing: -0.01em;
  color: var(--text-primary);
  min-width: 0;
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

// Slide-in: panel slides from right using the canonical panelSlide keyframe
// defined in interface.scss (translateX(48px) → 0). Leave transition handles unmount.
.thread-panel-enter-active {
  animation: panelSlide 0.4s var(--ease-out);
}

.thread-panel-leave-active {
  transition: opacity 200ms ease, transform 200ms ease;
}

.thread-panel-leave-to {
  opacity: 0;
  transform: translateX(48px);
}

@media (prefers-reduced-motion: reduce) {
  .thread-panel-enter-active {
    animation: none;
  }

  .thread-panel-leave-active {
    transition: none;
  }
}
</style>
