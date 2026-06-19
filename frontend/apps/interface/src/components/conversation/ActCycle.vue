<script setup lang="ts">
import { ref, computed, watch, onUnmounted } from 'vue';
import type { ActForm, ToolPill } from '../../stores/conversation';
import { useSessionStore } from '../../stores/session';

const props = defineProps<{ form: ActForm }>();

const session = useSessionStore();

// Wire stop button to the store's cancel action.
function onStop(): void {
  void session.requestStop();
}

// ── Live ticking clock ──────────────────────────────────────────────────────
// Drives the running-timer readout. Runs ONLY while this step is live (not
// collapsed) and at least one pill is still unresolved; torn down the instant
// everything resolves or the step is superseded, so settled turns cost nothing.
const now = ref(Date.now());
let timer: ReturnType<typeof setInterval> | null = null;

const hasRunning = computed(
  () => !props.form.collapsed && props.form.tools.some((t) => !t.resolved),
);

function stopClock(): void {
  if (timer) {
    clearInterval(timer);
    timer = null;
  }
}

watch(
  hasRunning,
  (running) => {
    if (running && !timer) {
      timer = setInterval(() => {
        now.value = Date.now();
      }, 100);
    } else if (!running) {
      stopClock();
    }
  },
  { immediate: true },
);

onUnmounted(stopClock);

/**
 * Seconds shown on a pill: the server/client-measured duration once resolved,
 * otherwise the live elapsed since the call started (ticks via `now`).
 */
function pillSeconds(pill: ToolPill): string {
  const ms = pill.resolved
    ? Math.max(0, pill.ms ?? 0)
    : Math.max(0, pill.startedAt ? now.value - pill.startedAt : 0);
  return (ms / 1000).toFixed(1);
}
</script>

<template>
  <div class="act-cycle" :class="{ 'act-cycle--collapsed': form.collapsed }">
    <!-- Live working anchor: pulsing logo + stop button. Dropped once the step
         is superseded (collapsed) — only the summaries remain. -->
    <div v-if="!form.collapsed" class="act-row">
      <span class="act-logo" />
      <button
        class="act-stop-btn"
        aria-label="Stop and undo"
        title="Stop & undo"
        type="button"
        @click="onStop"
      >
        <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor">
          <path d="M4.5 2L1 5.5L4.5 9V6.5H10a3.5 3.5 0 0 1 0 7H7v2h3a5.5 5.5 0 0 0 0-11H4.5V2Z" />
        </svg>
      </button>
    </div>

    <!-- Collapsed: summary-only lines. The red tool-name pill and the running
         timer are dropped — only the act summary survives (Dylan's spec). -->
    <div v-if="form.collapsed" class="act-summaries">
      <span
        v-for="pill in form.tools"
        :key="pill.id"
        class="act-tool__summary act-tool__summary--collapsed"
      >
        {{ pill.summary || pill.name }}
      </span>
    </div>

    <!-- Live: running / done pills with name + ticking timer (green on done). -->
    <div v-else class="act-tools">
      <div
        v-for="pill in form.tools"
        :key="pill.id"
        class="act-tool"
        :class="{
          'act-tool--running': !pill.resolved,
          'act-tool--done':    pill.resolved && pill.ok,
          'act-tool--error':   pill.resolved && !pill.ok,
        }"
        :data-call-id="pill.id"
      >
        <template v-if="pill.summary">
          <span class="act-tool__label">
            <span class="act-tool__name">{{ pill.name }}</span>
            <span class="act-tool__summary">— {{ pill.summary }}</span>
          </span>
        </template>
        <span v-else class="act-tool__name">{{ pill.name }}</span>

        <span class="act-tool__status">
          <template v-if="!pill.resolved">
            <span class="act-spinner" />
            <span class="act-tool__elapsed">{{ pillSeconds(pill) }}s</span>
          </template>
          <template v-else-if="pill.ok">{{ pillSeconds(pill) }}s</template>
          <template v-else>error</template>
        </span>
      </div>
    </div>
  </div>
</template>
