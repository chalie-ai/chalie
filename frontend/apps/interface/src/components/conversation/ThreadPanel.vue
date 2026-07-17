<!-- Slide-over thread panel: a focused, full-height view of one thread.
     Registers its body as a DOM-contract surface (D14) and fetches its own
     turn via REST — no buffer read for rendering. -->
<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue';
import { ArrowLeft } from '@lucide/vue';
import { useSessionStore } from '../../stores/session';
import { conversation as convoApi } from '../../api/conversation';
import { registerSurface, unregisterSurface, clearSurfaceContainer, upsertTurnToSurfaces } from '../../utils/turnDom';
import TurnView from './TurnView.vue';
import InputDock from '../layout/InputDock.vue';

const PANEL_SURFACE_ID = 'thread-panel';

const session = useSessionStore();

const open = computed(() => session.panelThreadId != null);

const heading = ref('Thread');
const hydrated = ref(false);

// `bodyRef` is the scrollable wrapper (kept for scrollTop pinning);
// `turnsRef` is the DEDICATED surface container — kept separate from the
// loader's own Vue-rendered node so imperative host divs (turnDom.mount)
// never share a parent with vdom-owned siblings.
const bodyRef = ref<HTMLElement | null>(null);
const turnsRef = ref<HTMLElement | null>(null);

function close(): void {
  session.closeThreadPanel();
}

/** Tear down the panel's surface registration and unmount its rendered turn. */
function _teardownSurface(): void {
  unregisterSurface(PANEL_SURFACE_ID);
  if (turnsRef.value) clearSurfaceContainer(turnsRef.value);
  hydrated.value = false;
}

/** (Re-)register the surface for the currently-open turn and fetch it. */
async function _openTurn(turnId: number, type: string): Promise<void> {
  _teardownSurface();
  await nextTick(); // let `v-if="open"` mount <aside> so turnsRef exists.
  if (!turnsRef.value || session.panelThreadId !== turnId || session.panelType !== type) return;

  registerSurface({
    id: PANEL_SURFACE_ID,
    type,
    container: turnsRef.value,
    component: TurnView,
    props: { canReply: false, fullThread: true },
    accepts: (id) => id === turnId,
  });

  session.threadExpanding = true;
  try {
    const block = await convoApi.thread(turnId, type);
    if (session.panelThreadId !== turnId || session.panelType !== type) return; // superseded
    // turn_id is only unique PER TYPE — trust the block's own authoritative
    // type over whatever this fetch was requested with, loudly on disagreement.
    if (block.type !== type) {
      console.warn(
        '[ThreadPanel] refetched turn', turnId, 'came back as type', block.type,
        'but was requested as', type, '— trusting the fetched type',
      );
    }
    heading.value = block.gist || block.preview || 'Thread';
    upsertTurnToSurfaces(block, block.type);
    hydrated.value = true;
  } catch {
    if (session.panelThreadId !== turnId || session.panelType !== type) return; // superseded
    hydrated.value = true; // stop the spinner — nothing more is coming
    session.errorMessage = 'Failed to load this thread';
  } finally {
    if (session.panelThreadId === turnId && session.panelType === type) {
      session.threadExpanding = false;
    }
  }
}

watch(
  () => [session.panelThreadId, session.panelType] as const,
  ([turnId, type]) => {
    if (turnId == null) {
      _teardownSurface();
      return;
    }
    void _openTurn(turnId, type);
  },
);

// Follow live reply growth: pin the body to its bottom whenever the open
// turn re-renders (turnDom's 'turn-upserted' signal, dispatched after every
// DOM write — replaces the old watch on the buffer's block).
function onTurnUpserted(e: Event): void {
  if (!open.value) return;
  const detail = (e as CustomEvent<{ turnId: number; type: string }>).detail;
  if (detail.turnId !== session.panelThreadId || detail.type !== session.panelType) return;
  nextTick(() => {
    const el = bodyRef.value;
    if (el) el.scrollTop = el.scrollHeight;
  });
}

function onKeydown(e: KeyboardEvent): void {
  if (e.key === 'Escape' && open.value) close();
}

onMounted(() => {
  document.addEventListener('keydown', onKeydown);
  document.addEventListener('turn-upserted', onTurnUpserted);
  if (session.panelThreadId != null) void _openTurn(session.panelThreadId, session.panelType);
});

onBeforeUnmount(() => {
  document.removeEventListener('keydown', onKeydown);
  document.removeEventListener('turn-upserted', onTurnUpserted);
  _teardownSurface();
});
</script>

<template>
  <Transition name="thread-panel">
    <aside
      v-if="open"
      class="thread-panel"
      role="dialog"
      aria-modal="true"
      :aria-label="heading"
      :data-dock-scope="session.panelThreadId"
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
        <span class="thread-panel__title">{{ heading }}</span>
      </header>

      <div ref="bodyRef" class="thread-panel__body">
        <div v-if="!hydrated" class="thread-panel__loader">
          <output class="thread-panel__spinner" aria-label="Loading thread" />
        </div>
        <div ref="turnsRef" class="thread-panel__turns" />
      </div>

      <InputDock
        v-if="session.panelThreadId != null"
        dock-id="thread_view_dock"
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
  background: var(--scrim-panel-main);
  // Published for .user-text--clamped::after (conversation.scss) — rows in
  // this panel sit on the translucent scrim, not the page background.
  --row-fade-bg: var(--scrim-panel-main);
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
