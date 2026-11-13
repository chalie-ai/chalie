<!-- One dock row per pending ask no live line on screen shows (a scheduled
     task, or a thread whose panel is closed). Open shows the turn in the
     thread panel; the ask is answered on that turn's live line. A browser
     hand-off gets a row while the browser panel is closed, or while it is
     open on behalf of another turn; its Open shows the tab in the browser
     panel, where Done answers it. -->
<script setup lang="ts">
import { computed, onUnmounted, ref, watch } from 'vue';
import { useSessionStore } from '../../stores/session';
import { usePermissionsStore, type PermissionRequest } from '../../stores/permissions';
import { useBrowserStore } from '../../stores/browser';
import { actionLabel, originLabel, waitedSeconds } from '../../utils/permissionAsk';

const session = useSessionStore();
const permissions = usePermissionsStore();
const browser = useBrowserStore();
const browserAway = computed(() =>
  browser.awayFrom({ turnId: session.panelThreadId, type: session.panelType }),
);

// Ticks the "waiting for you" ages only while a row is showing.
const now = ref(Date.now());
let timer: ReturnType<typeof setInterval> | null = null;

function stopClock(): void {
  if (timer) clearInterval(timer);
  timer = null;
}

watch(
  () => permissions.offScreen.length + browserAway.value.length,
  (n) => {
    if (n > 0 && !timer) {
      timer = setInterval(() => {
        now.value = Date.now();
      }, 1000);
    } else if (n === 0) {
      stopClock();
    }
  },
  { immediate: true },
);

onUnmounted(stopClock);

function open(ask: PermissionRequest): void {
  session.openThreadPanel(ask.origin.turn_id, ask.origin.type);
}
</script>

<template>
  <!-- Held through the conversation's first load: until its turns render, an
       ask their live lines are about to claim would flash here first. -->
  <div class="offscreen-asks" aria-live="polite">
    <div v-for="ask in session.historyLoading ? [] : permissions.offScreen" :key="ask.request_id" class="activity">
      <span class="activity__lead">
        <span class="activity__mark" />
        <span class="activity__origin">{{ originLabel(ask.origin) }} · #{{ ask.origin.turn_id }} ·</span>
        <span class="activity__label">{{ ask.summary || actionLabel(ask.action_id) }}</span>
      </span>
      <span class="activity__controls">
        <span class="activity__wait">· waiting for you · {{ waitedSeconds(ask.asked_at, now) }}s</span>
        <span v-if="ask.summary" class="activity__ask">{{ actionLabel(ask.action_id) }}</span>
        <button type="button" class="activity__stop" @click="open(ask)">Open</button>
      </span>
    </div>
    <div v-for="ask in session.historyLoading ? [] : browserAway" :key="ask.handoff_id" class="activity">
      <span class="activity__lead">
        <span class="activity__mark" />
        <span class="activity__origin">{{ originLabel(ask.origin) }} · #{{ ask.origin.turn_id }} ·</span>
        <span class="activity__label">{{ ask.reason || 'Chalie needs you in the browser' }}</span>
      </span>
      <span class="activity__controls">
        <span class="activity__wait">· waiting for you · {{ waitedSeconds(ask.asked_at, now) }}s</span>
        <span class="activity__ask">Browser</span>
        <button type="button" class="activity__stop" @click="browser.openPanel(ask.tab_id)">Open</button>
      </span>
    </div>
  </div>
</template>
