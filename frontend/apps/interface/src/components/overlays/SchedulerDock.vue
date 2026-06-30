<script setup lang="ts">
/**
 * Scheduler dock — slide-out panel listing active prompt-schedule threads.
 * Each row shows the thread's gist/preview, a blinker driven by the schedule
 * feed's threadPhase, and opens the thread panel on click.
 *
 * Open/close is driven by session.schedulerDockOpen. Polling runs only while open.
 */
import { ref, watch, onMounted, onBeforeUnmount } from 'vue';
import { X } from '@lucide/vue';
import { useSessionStore } from '../../stores/session';
import { scheduler } from '../../api/scheduler';
import type { SchedulerTurn } from '../../api/scheduler';
import { useConversationFeed } from '../../composables/useConversationFeed';

const POLL_INTERVAL_MS = 60_000;

const session = useSessionStore();
const feed = useConversationFeed('schedule');

// ── Local state ───────────────────────────────────────────────────────────────

const turns = ref<SchedulerTurn[]>([]);
let pollTimer: ReturnType<typeof setInterval> | null = null;

const drawerRef = ref<HTMLElement | null>(null);
const scrimRef = ref<HTMLElement | null>(null);

// ── Polling ───────────────────────────────────────────────────────────────────

async function loadTurns(): Promise<void> {
  try {
    turns.value = await scheduler.turns();
  } catch {
    /* keep stale rows rather than blanking on transient error */
  }
}

function startPolling(): void {
  void loadTurns();
  pollTimer = setInterval(() => { void loadTurns(); }, POLL_INTERVAL_MS);
}

function stopPolling(): void {
  if (pollTimer !== null) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

// ── DOM open/close (mirrors TaskDrawer) ──────────────────────────────────────

function openDrawerDom(): void {
  if (!scrimRef.value || !drawerRef.value) return;
  scrimRef.value.classList.remove('hidden');
  drawerRef.value.classList.remove('hidden');
  requestAnimationFrame(() => {
    drawerRef.value?.classList.add('open');
  });
}

function closeDrawerDom(): void {
  const drawer = drawerRef.value;
  const scrim = scrimRef.value;
  if (!drawer) return;
  drawer.classList.remove('open');
  scrim?.classList.add('hidden');
  drawer.addEventListener(
    'transitionend',
    () => {
      if (!drawer.classList.contains('open')) drawer.classList.add('hidden');
    },
    { once: true },
  );
}

watch(
  () => session.schedulerDockOpen,
  (open) => {
    if (open) {
      openDrawerDom();
      startPolling();
    } else {
      closeDrawerDom();
      stopPolling();
    }
  },
);

// ── Keyboard ──────────────────────────────────────────────────────────────────

function handleKeydown(e: KeyboardEvent): void {
  if (e.key === 'Escape' && session.schedulerDockOpen) session.closeSchedulerDock();
}

onMounted(() => document.addEventListener('keydown', handleKeydown));
onBeforeUnmount(() => {
  document.removeEventListener('keydown', handleKeydown);
  stopPolling();
});

// ── Row interaction ───────────────────────────────────────────────────────────

function openTurn(turn: SchedulerTurn): void {
  session.openThreadPanel(turn.turn_id, 'schedule');
  session.closeSchedulerDock();
}
</script>

<template>
  <!-- Scrim -->
  <div
    class="sched-dock__scrim hidden"
    ref="scrimRef"
    aria-hidden="true"
    @click="session.closeSchedulerDock()"
  ></div>

  <!-- Slide-out panel -->
  <aside
    ref="drawerRef"
    class="sched-dock hidden"
    role="complementary"
    aria-label="Schedules"
  >
    <div class="sched-dock__header">
      <h2 class="sched-dock__title">Schedules</h2>
      <button
        class="btn-icon sched-dock__close"
        aria-label="Close schedules panel"
        @click="session.closeSchedulerDock()"
      >
        <X :size="16" aria-hidden="true" />
      </button>
    </div>

    <div class="sched-dock__list">
      <p v-if="!turns.length" class="sched-dock__empty">No active schedules.</p>

      <button
        v-for="turn in turns"
        :key="turn.turn_id"
        class="sched-dock__row"
        :class="feed.threadPhase(turn.turn_id) ? `sched-dock__row--${feed.threadPhase(turn.turn_id)}` : ''"
        @click="openTurn(turn)"
      >
        <span class="sched-dock__row-top">
          <span class="sched-dock__row-label">{{ turn.gist || turn.preview }}</span>
          <span
            v-if="feed.threadPhase(turn.turn_id)"
            class="task-drawer__thread-dot"
            :class="feed.threadPhase(turn.turn_id) ?? ''"
            aria-hidden="true"
          />
        </span>
        <span v-if="turn.recurrence" class="sched-dock__row-sub">{{ turn.recurrence }}</span>
      </button>
    </div>
  </aside>
</template>

<style scoped lang="scss">
// ── Scrim ─────────────────────────────────────────────────────────────────────

.sched-dock__scrim {
  position: fixed;
  inset: 0;
  background: var(--overlay-scrim, rgba(0, 0, 0, 0.35));
  z-index: 199;
  opacity: 1;
  transition: opacity 0.2s ease;

  &.hidden {
    display: none;
  }
}

// ── Panel ─────────────────────────────────────────────────────────────────────

.sched-dock {
  position: fixed;
  top: 0;
  right: 0;
  height: 100%;
  width: 320px;
  max-width: 90vw;
  background: var(--bg-2);
  border-left: 1px solid var(--border);
  box-shadow: -4px 0 24px var(--shadow, rgba(0, 0, 0, 0.15));
  z-index: 200;
  display: flex;
  flex-direction: column;
  transform: translateX(100%);
  transition: transform 0.25s ease;
  overflow: hidden;

  &.open {
    transform: translateX(0);
  }

  &.hidden {
    display: none;
  }
}

.sched-dock__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 16px 12px;
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
}

.sched-dock__title {
  margin: 0;
  font-size: 15px;
  font-weight: 600;
  color: var(--text-primary);
}

.sched-dock__close {
  color: var(--text-secondary);

  &:hover {
    color: var(--text-primary);
  }
}

// ── List ──────────────────────────────────────────────────────────────────────

.sched-dock__list {
  flex: 1;
  overflow-y: auto;
  padding: 8px 0;
}

.sched-dock__empty {
  font-size: 13px;
  color: var(--text-secondary);
  padding: 10px 16px;
  font-style: italic;
  margin: 0;
}

// ── Row ───────────────────────────────────────────────────────────────────────

.sched-dock__row {
  display: flex;
  flex-direction: column;
  gap: 4px;
  width: 100%;
  text-align: left;
  background: transparent;
  border: none;
  border-left: 2px solid transparent;
  padding: 10px 16px;
  cursor: pointer;

  &:hover {
    background: var(--surface-hover, rgba(128, 128, 128, 0.06));
  }
}

.sched-dock__row--working { border-left-color: var(--status-main); }
.sched-dock__row--done    { border-left-color: var(--cyan); }

.sched-dock__row-top {
  display: flex;
  align-items: center;
  gap: 8px;
}

.sched-dock__row-label {
  flex: 1;
  min-width: 0;
  font-size: 13px;
  font-weight: 600;
  color: var(--text-primary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.sched-dock__row-sub {
  font-size: 11.5px;
  color: var(--text-secondary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

// Blinker dot — same tokens as TaskDrawer's .task-drawer__thread-dot.
// pulseV is defined globally in interface.scss.
.task-drawer__thread-dot {
  flex-shrink: 0;
  width: 8px;
  height: 8px;
  border-radius: 50%;

  &.working {
    background: var(--status-main);
    box-shadow: 0 0 10px color-mix(in oklab, var(--status-main) 60%, transparent);
    animation: pulseV 1.4s ease-in-out infinite;
  }

  &.done {
    background: var(--cyan);
    box-shadow: 0 0 8px color-mix(in oklab, var(--cyan) 50%, transparent);
  }
}
</style>
