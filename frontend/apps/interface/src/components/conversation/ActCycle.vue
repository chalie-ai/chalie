<script setup lang="ts">
import { computed, onUnmounted, ref, watch } from 'vue';
import { Bot, Undo2 } from '@lucide/vue';
import { readDomContext } from '../../utils/domContext';
import { lastUserText } from '../../utils/turnDom';
import type { LiveToolPill } from '../../utils/liveActTrail';
import { useSessionStore } from '../../stores/session';
import { delegatePillAttrs } from '../../composables/useDelegatePill';

const props = withDefaults(
  defineProps<{
    pills: LiveToolPill[];
    /** False where a stop interrupts without handing anything back to undo (a
     *  delegate's transcript). */
    undoable?: boolean;
  }>(),
  { undoable: true },
);

const session = useSessionStore();
const rootRef = ref<HTMLElement | null>(null);

async function onStop(): Promise<void> {
  const { turnId, type, channel, dockScope } = readDomContext(rootRef.value);
  if (turnId == null) return; // Guard: never fire a stop without a target
  // The restore text (D6) must come from the SAME rendered copy this stop
  // button belongs to — the same turn_id can render different row sets on
  // different surfaces (see lastUserText's own doc comment).
  const turnHost = rootRef.value?.closest<HTMLElement>('[data-turn-id]') ?? null;
  const restoreText = turnHost ? lastUserText(turnHost) : '';
  const target = channel != null ? { channel, turn_id: turnId } : turnId;
  await session.requestStop(target, type, dockScope, restoreText);
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
  <div ref="rootRef" class="act-cycle">
    <!-- Live working anchor (logo + stop). -->
    <div class="act-row">
      <span class="act-logo" />
      <button
        class="act-stop-btn"
        :aria-label="undoable ? 'Stop and undo' : 'Stop subagent'"
        :title="undoable ? 'Stop & undo' : 'Stop subagent'"
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
      <component
        :is="pill.delegate ? 'button' : 'div'"
        v-for="pill in pills"
        :key="pill.id"
        v-bind="delegatePillAttrs(pill.delegate)"
        class="act-tool"
        :class="{
          'act-tool--running': !pill.resolved,
          'act-tool--done': pill.resolved && pill.ok,
          'act-tool--error': pill.resolved && !pill.ok,
        }"
        :data-call-id="pill.id"
        :data-transcript-row-id="pill.transcriptRowId"
      >
        <span class="act-tool__label">
          <span class="act-tool__name">
            <Bot v-if="pill.delegate" class="delegate-pill__icon" :size="14" aria-hidden="true" />{{
              pill.name
            }}
          </span>
          <span v-if="pill.summary" class="act-tool__summary">— {{ pill.summary }}</span>
        </span>

        <span class="act-tool__status">
          <template v-if="!pill.resolved">
            <span class="act-tool__elapsed">{{ pillSeconds(pill) }}s</span>
          </template>
          <template v-else-if="pill.ok">{{ pillSeconds(pill) }}s</template>
          <template v-else>error</template>
        </span>
      </component>
    </div>
  </div>
</template>
