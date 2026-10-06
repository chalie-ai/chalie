<!-- Slide-over thread panel: a focused, full-height view of one thread, or a
     watch-only view of one delegate (subagent) turn's transcript (no reply,
     only a stop while it runs).
     Registers its body as a DOM-contract surface (D14) and fetches its own
     turn via REST — no buffer read for rendering. -->
<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue';
import { ArrowLeft, Bot } from '@lucide/vue';
import type { DelegateRef } from '@chalie/shared';
import { useSessionStore } from '../../stores/session';
import { conversation as convoApi } from '../../api/conversation';
import {
  registerSurface,
  unregisterSurface,
  clearSurfaceContainer,
  upsertDelegateTurn,
  upsertTurnToSurfaces,
} from '../../utils/turnDom';
import { delegateName } from '../../utils/delegateChannel';
import TurnView from './TurnView.vue';
import InputDock from '../layout/InputDock.vue';

const PANEL_SURFACE_ID = 'thread-panel';

const session = useSessionStore();

const delegateMode = computed(() => session.panelDelegate != null);
const open = computed(() => session.panelThreadId != null || delegateMode.value);

const heading = ref('Thread');
const hydrated = ref(false);
/** The open delegate turn came back with no rows: its transcript is gone. */
const expired = ref(false);

// `bodyRef` is the scrollable wrapper (kept for scrollTop pinning);
// `turnsRef` is the DEDICATED surface container — kept separate from the
// loader's own Vue-rendered node so imperative host divs (turnDom.mount)
// never share a parent with vdom-owned siblings.
const bodyRef = ref<HTMLElement | null>(null);
const turnsRef = ref<HTMLElement | null>(null);
const backRef = ref<HTMLButtonElement | null>(null);
/** What had focus when the panel opened; it gets focus back on close. */
let opener: HTMLElement | null = null;

function close(): void {
  session.closeThreadPanel();
}

/** Tear down the panel's surface registration and unmount its rendered turn. */
function _teardownSurface(): void {
  unregisterSurface(PANEL_SURFACE_ID);
  if (turnsRef.value) clearSurfaceContainer(turnsRef.value);
  hydrated.value = false;
  expired.value = false;
}

/** (Re-)register the surface for the currently-open turn and fetch it. */
async function _openTurn(turnId: number, type: string): Promise<void> {
  _teardownSurface();
  await nextTick(); // let `v-if="open"` mount <aside> so turnsRef exists.
  if (!turnsRef.value || session.panelThreadId !== turnId || session.panelType !== type) return;
  backRef.value?.focus();

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

/** True while `target` is still the delegate turn the panel is open on. */
function isOpenDelegate(target: DelegateRef): boolean {
  const current = session.panelDelegate;
  return current != null && current.channel === target.channel && current.turn_id === target.turn_id;
}

/** (Re-)register the surface for a delegate turn — addressed by its channel,
 *  never by a type — and fetch it. */
async function _openDelegate(target: DelegateRef): Promise<void> {
  _teardownSurface();
  heading.value = delegateName(target.channel);
  await nextTick(); // let `v-if="open"` mount <aside> so turnsRef exists.
  if (!turnsRef.value || !isOpenDelegate(target)) return;
  backRef.value?.focus();

  registerSurface({
    id: PANEL_SURFACE_ID,
    channel: target.channel,
    container: turnsRef.value,
    component: TurnView,
    props: { canReply: false, fullThread: true },
    accepts: (id) => id === target.turn_id,
  });

  try {
    const block = await convoApi.delegateThread(target.turn_id, target.channel);
    if (!isOpenDelegate(target)) return; // superseded
    hydrated.value = true;
    expired.value = block.messages.length === 0;
    if (!expired.value) upsertDelegateTurn(block);
  } catch (err) {
    if (!isOpenDelegate(target)) return; // superseded
    console.warn('[ThreadPanel] delegate turn', target.turn_id, 'on', target.channel, 'failed to load', err);
    hydrated.value = true; // stop the spinner — nothing more is coming
    session.errorMessage = 'Failed to load this transcript';
  }
}

watch(open, (isOpen) => {
  if (isOpen) {
    opener = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    return;
  }
  if (opener?.isConnected) opener.focus();
  opener = null;
});

watch(
  () => [session.panelThreadId, session.panelType, session.panelDelegate] as const,
  ([turnId, type, delegate]) => {
    if (delegate != null) {
      void _openDelegate(delegate);
      return;
    }
    if (turnId == null) {
      _teardownSurface();
      return;
    }
    void _openTurn(turnId, type);
  },
);

/** True when a 'turn-upserted' detail names the turn this panel is open on —
 *  a delegate turn by its channel, a thread by its type. */
function isPanelTurn(detail: { turnId: number; type?: string; channel?: string }): boolean {
  const target = session.panelDelegate;
  if (target != null) return detail.channel === target.channel && detail.turnId === target.turn_id;
  return detail.turnId === session.panelThreadId && detail.type === session.panelType;
}

// Follow live reply growth: pin the body to its bottom whenever the open
// turn re-renders (turnDom's 'turn-upserted' signal, dispatched after every
// DOM write — replaces the old watch on the buffer's block).
function onTurnUpserted(e: Event): void {
  if (!open.value) return;
  const detail = (e as CustomEvent<{ turnId: number; type?: string; channel?: string }>).detail;
  if (!isPanelTurn(detail)) return;
  expired.value = false;
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
  if (session.panelDelegate != null) void _openDelegate(session.panelDelegate);
  else if (session.panelThreadId != null) void _openTurn(session.panelThreadId, session.panelType);
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
      :class="{ 'thread-panel--delegate': delegateMode }"
      role="dialog"
      aria-modal="true"
      :aria-label="delegateMode ? `${heading} subagent transcript` : heading"
      :data-dock-scope="session.panelThreadId"
    >
      <header class="thread-panel__header">
        <button
          ref="backRef"
          class="thread-panel__back"
          type="button"
          aria-label="Back to conversation"
          @click="close"
        >
          <ArrowLeft :size="16" />
          <span>Chalie</span>
        </button>
        <Bot v-if="delegateMode" class="thread-panel__fork-glyph thread-panel__bot-glyph" :size="16" aria-hidden="true" />
        <svg
          v-else
          class="thread-panel__fork-glyph"
          width="16"
          height="16"
          viewBox="0 0 24 24"
          fill="none"
          stroke="var(--pink-text)"
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
        <div v-else-if="expired" class="thread-panel__expired">
          <p class="thread-panel__expired-title">Transcript expired</p>
          <p class="thread-panel__expired-note">This subagent's transcript is no longer kept.</p>
        </div>
        <div ref="turnsRef" class="thread-panel__turns" />
      </div>

      <!-- Permission cards for the turn this panel shows: PermissionStack.vue
           teleports them here, in flow above this dock, while the panel is open
           on their turn. The target lives only with the open panel. A delegate
           transcript takes no reply: no dock, and its cards stay on the main
           stack since no thread is open. -->
      <div id="permStackPanel" class="permission-stack permission-stack--panel"></div>

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
  background: var(--surface);
  overflow: hidden;
}

.thread-panel__header {
  flex-shrink: 0;
  display: flex;
  align-items: center;
  gap: 11px;
  height: 46px;
  padding: 0 26px;
}

.thread-panel__back {
  display: flex;
  align-items: center;
  gap: 7px;
  margin-left: -4px;
  padding: 5px 9px 5px 6px;
  background: none;
  color: var(--muted);
  font:
    500 var(--fs-body) var(--font-ui);
  cursor: pointer;
  transition:
    color var(--dur-1) var(--ease-out),
    background-color var(--dur-1) var(--ease-out),
    translate var(--dur-1) var(--ease-out);
}

.thread-panel__back:hover {
  color: var(--text);
  background: var(--surface-2);
}

.thread-panel__back:active {
  translate: 0 1px;
}

.thread-panel__fork-glyph {
  flex-shrink: 0;
}

.thread-panel__bot-glyph {
  color: var(--pink-text);
}

// A delegate transcript is a side read, not a working thread: narrower, so the
// conversation it came from stays in view, and full width on small screens.
.thread-panel--delegate {
  width: min(760px, 95%);
}

@media (max-width: 640px) {
  .thread-panel--delegate {
    width: 100%;
  }
}

.thread-panel__title {
  font:
    600 var(--fs-title) var(--font-ui);
  letter-spacing: -0.01em;
  color: var(--text);
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

.thread-panel__expired {
  padding: 32px 26px;
  text-align: center;
}

.thread-panel__expired-title {
  margin: 0 0 6px;
  font-weight: 600;
  color: var(--text);
}

.thread-panel__expired-note {
  margin: 0;
  font-size: var(--fs-body);
  color: var(--muted);
}

.thread-panel__spinner {
  width: 20px;
  height: 20px;
  border: 2px solid var(--line);
  border-top-color: var(--pink-text);
  animation: spin 0.7s linear infinite;
}

// The panel slides in on open and plays the same slide backwards on close.
.thread-panel-enter-active,
.thread-panel-leave-active {
  animation: slide-in var(--dur-3) var(--ease-out);
}

.thread-panel-leave-active {
  animation-direction: reverse;
  animation-fill-mode: forwards;
}
</style>
