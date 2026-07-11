<script setup lang="ts">
/**
 * Scheduler dock — slide-out panel listing active prompt-schedule threads.
 * Each row shows the thread's gist/preview, a blinker driven by the schedule
 * feed's threadPhase, and opens the thread panel on click.
 *
 * Open/close is driven by session.schedulerDockOpen. Polling runs only while open.
 */
import { onBeforeUnmount, onMounted, ref, watch } from 'vue';
import { ConfigType, describeCron } from '@chalie/shared';
import { useSessionStore } from '../../stores/session';
import type { SchedulerTurn } from '../../api/scheduler';
import { scheduler } from '../../api/scheduler';
import { threadPhase } from '../../utils/threadActivity';
import SideDrawer from './SideDrawer.vue';

const POLL_INTERVAL_MS = 60_000;

const session = useSessionStore();

// ── Local state ───────────────────────────────────────────────────────────────

const turns = ref<SchedulerTurn[]>([]);
let pollTimer: ReturnType<typeof setInterval> | null = null;

// Scheduled turns don't render on the spine, so `threadPhase` reads DOM
// contract state globally (any rendered copy, e.g. an open thread panel) —
// bumped on every 'turn-state-changed' so the dock's dots/borders stay live.
const activityTick = ref(0);
function bumpActivity(): void {
  activityTick.value++;
}
function phase(turn: SchedulerTurn): 'working' | 'done' | null {
  void activityTick.value;
  return threadPhase(turn.turn_id, ConfigType.SCHEDULED);
}

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
  pollTimer = setInterval(() => {
    void loadTurns();
  }, POLL_INTERVAL_MS);
}

function stopPolling(): void {
  if (pollTimer !== null) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

// Polling runs only while the dock is open.
watch(
  () => session.schedulerDockOpen,
  (open) => {
    if (open) startPolling();
    else stopPolling();
  },
);

onMounted(() => {
  document.addEventListener('turn-state-changed', bumpActivity);
});
onBeforeUnmount(() => {
  document.removeEventListener('turn-state-changed', bumpActivity);
  stopPolling();
});

// ── Row interaction ───────────────────────────────────────────────────────────

function openTurn(turn: SchedulerTurn): void {
  session.openThreadPanel(turn.turn_id, ConfigType.SCHEDULED);
  session.closeSchedulerDock();
}

/** Cadence sub-label for a dock row, derived from the schedule's cron fields. */
function cadence(turn: SchedulerTurn): string {
  return describeCron(turn.minute, turn.hour, turn.day, turn.month, turn.weekday);
}
</script>

<template>
  <SideDrawer
    :open="session.schedulerDockOpen"
    title="Schedules"
    @close="session.closeSchedulerDock()"
  >
    <p v-if="!turns.length" class="sched-dock__empty">No active schedules.</p>

    <button
      v-for="turn in turns"
      :key="turn.turn_id"
      class="sched-dock__row"
      :class="phase(turn) ? `sched-dock__row--${phase(turn)}` : ''"
      @click="openTurn(turn)"
    >
      <span class="sched-dock__row-top">
        <span class="sched-dock__row-label">{{ turn.gist || turn.preview }}</span>
        <span
          v-if="phase(turn)"
          class="thread-activity-dot"
          :class="phase(turn) ?? ''"
          aria-hidden="true"
        />
      </span>
      <span class="sched-dock__row-sub">{{ cadence(turn) }}</span>
    </button>
  </SideDrawer>
</template>

<style scoped lang="scss">
// ── Empty state ───────────────────────────────────────────────────────────────

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

.sched-dock__row--working {
  border-left-color: var(--status-main);
}
.sched-dock__row--done {
  border-left-color: var(--cyan);
}

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
</style>
