<!-- Slide-over thread panel: a focused, full-height view of one thread. -->
<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue';
import { ArrowLeft } from '@lucide/vue';
import { useSessionStore } from '../../stores/session';
import { useConversationFeed } from '../../composables/useConversationFeed';
import TurnView from './TurnView.vue';
import InputDock from '../layout/InputDock.vue';

const session = useSessionStore();
const feed = computed(() => useConversationFeed(session.panelType));

const open = computed(() => session.panelThreadId != null);

const block = computed(() =>
  session.panelThreadId != null ? (feed.value.blocks[session.panelThreadId] ?? null) : null,
);

const heading = computed(() => block.value?.gist || block.value?.preview || 'Thread');
const showLoader = computed(() => session.threadExpanding && block.value == null);

const bodyRef = ref<HTMLElement | null>(null);

function close(): void {
  session.closeThreadPanel();
}

// Follow live reply growth: pin the body to its bottom whenever the open
// thread's block changes. flush:'post' lets it measure settled height.
watch(
  block,
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
    <aside v-if="open" class="thread-panel" role="dialog" aria-modal="true" :aria-label="heading">
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
        <TurnView v-else-if="block" :block="block" :can-reply="false" :type="session.panelType" />
      </div>

      <InputDock
        v-if="session.panelThreadId != null"
        :turn-id="session.panelThreadId"
        :type="session.panelType"
      />
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
  z-index: 120;
  display: flex;
  flex-direction: column;
  background: var(--scrim-panel-thread);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
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
  font:
    500 13px Inter,
    system-ui,
    sans-serif;
  cursor: pointer;
  transition:
    color var(--duration-fast),
    background var(--duration-fast);
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
  font:
    600 14px Inter,
    system-ui,
    sans-serif;
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
  animation: spin 0.7s linear infinite;
}

.thread-panel-enter-active {
  animation: panelSlide 0.4s var(--ease-out);
}

.thread-panel-leave-active {
  transition:
    opacity 200ms ease,
    transform 200ms ease;
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
