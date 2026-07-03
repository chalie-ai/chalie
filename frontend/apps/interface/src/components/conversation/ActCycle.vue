<script setup lang="ts">
import { computed, onUnmounted, ref, watch } from 'vue';
import { Undo2 } from '@lucide/vue';
import { ConfigType } from '@chalie/shared';
import type { LiveToolPill } from '../../composables/useConversationFeed';
import { useSessionStore } from '../../stores/session';

const props = withDefaults(
  defineProps<{ pills: LiveToolPill[]; turnId: number | null; type?: string }>(),
  { type: ConfigType.USER },
);

const session = useSessionStore();

async function onStop(): Promise<void> {
  await session.requestStop(props.turnId, props.type);
}

// Live timer: ticks ONLY while a pill is unresolved.
const now = ref(Date.now());
let timer: ReturnType<typeof setInterval> | null = null;

const hasRunning = computed(() => props.pills.some((p) => !p.resolved));

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

function pillSeconds(pill: LiveToolPill): string {
  const ms = pill.resolved
    ? Math.max(0, pill.ms ?? 0)
    : Math.max(0, pill.startedAt ? now.value - pill.startedAt : 0);
  return (ms / 1000).toFixed(1);
}
</script>

<template>
  <div class="act-cycle">
    <!-- Live working anchor (logo + stop). -->
    <div class="act-row">
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

    <!-- Running / done pills with name + ticking timer. Before the first pill
         lands, the bare group is the "thinking…" anchor. -->
    <div class="act-tools">
      <span v-if="!pills.length" class="act-placeholder">thinking…</span>
      <div
        v-for="pill in pills"
        :key="pill.id"
        class="act-tool"
        :class="{
          'act-tool--running': !pill.resolved,
          'act-tool--done': pill.resolved && pill.ok,
          'act-tool--error': pill.resolved && !pill.ok,
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
