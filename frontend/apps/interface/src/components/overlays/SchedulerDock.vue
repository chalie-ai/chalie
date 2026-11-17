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
// bumped on every 'turn-state-changed' so the dock's dots stay live.
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
    <p v-if="!turns.length" class="drawer-empty">No active schedules.</p>

    <button
      v-for="turn in turns"
      :key="turn.turn_id"
      class="drawer-row"
      @click="openTurn(turn)"
    >
      <span class="drawer-row__top">
        <span class="drawer-row__label">{{ turn.gist || turn.preview }}</span>
        <span
          v-if="phase(turn)"
          class="thread-activity-dot"
          :class="phase(turn) ?? ''"
          aria-hidden="true"
        />
      </span>
      <span class="drawer-row__sub">{{ cadence(turn) }}</span>
    </button>
  </SideDrawer>
</template>
