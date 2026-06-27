<script setup lang="ts">
import { ref, computed, watch, onUnmounted } from 'vue';
import { Undo2 } from '@lucide/vue';
import type { ActForm, ToolPill } from '../../stores/conversation';
import { useSessionStore } from '../../stores/session';

const props = defineProps<{ form: ActForm }>();

const session = useSessionStore();

function onStop(): void {
  void session.requestStop();
}

// Live timer: ticks ONLY while the step is live and a pill is unresolved; torn
// down the instant everything resolves, so settled turns cost nothing.
const now = ref(Date.now());
let timer: ReturnType<typeof setInterval> | null = null;

const hasRunning = computed(
  () => !props.form.collapsed && props.form.tools.some((t) => !t.resolved),
);

function stopClock(): void {
  if (timer) clearInterval(timer);
  timer = null;
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

// Seconds on a pill: measured duration once resolved, else live elapsed.
function pillSeconds(pill: ToolPill): string {
  const ms = pill.resolved
    ? Math.max(0, pill.ms ?? 0)
    : Math.max(0, pill.startedAt ? now.value - pill.startedAt : 0);
  return (ms / 1000).toFixed(1);
}
</script>

<template>
  <div class="act-cycle" :class="{ 'act-cycle--collapsed': form.collapsed }">
    <!-- Live working anchor (logo + stop); dropped once superseded. -->
    <div v-if="!form.collapsed" class="act-row">
      <span class="act-logo" />
      <button
        class="act-stop-btn"
        aria-label="Stop and undo"
        title="Stop & undo"
        type="button"
        @click="onStop"
      >
        <Undo2 :size="14" />
      </button>
    </div>

    <!-- Collapsed: summary-only lines; tool-name pill and timer dropped. -->
    <div v-if="form.collapsed" class="act-summaries">
      <span
        v-for="pill in form.tools"
        :key="pill.id"
        class="act-tool__summary act-tool__summary--collapsed"
      >
        {{ pill.summary || pill.name }}
      </span>
    </div>

    <!-- Live: running / done pills with name + ticking timer (green on done).
         Before the first pill lands, the bare group is the "thinking…" anchor. -->
    <div v-else class="act-tools">
      <span v-if="!form.tools.length" class="act-placeholder">{{ form.placeholder || 'thinking…' }}</span>
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
        <span v-if="pill.summary" class="act-tool__label">
          <span class="act-tool__name">{{ pill.name }}</span>
          <span class="act-tool__summary">— {{ pill.summary }}</span>
        </span>
        <span v-else class="act-tool__name">{{ pill.name }}</span>

        <span class="act-tool__status">
          <template v-if="!pill.resolved">
            <span class="act-tool__elapsed">{{ pillSeconds(pill) }}s</span>
          </template>
          <template v-else-if="pill.ok">{{ pillSeconds(pill) }}s</template>
          <template v-else>error</template>
        </span>
      </div>
    </div>
  </div>
</template>
