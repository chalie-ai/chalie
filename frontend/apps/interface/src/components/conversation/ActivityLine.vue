<script setup lang="ts">
/**
 * ActivityLine — the running step of a still-working exchange: the same work a
 * settled row shows (ThinkingAndCalls), live — the stored row's thinking and
 * calls once a refetch returned it, then the act-trail pills running now — its
 * head carrying the turn's controls. Stop always; while the turn is parked on
 * a permission ask, the head names the ask, a ticking "· waiting for you · Ns"
 * note follows — aged from the server's `asked_at` stamp, so it survives a
 * reload — and the ask is answered on the head itself (Deny/Allow).
 */
import { computed, onUnmounted, ref, watch } from 'vue';
import { readDomContext } from '../../utils/domContext';
import { lastUserText } from '../../utils/turnDom';
import type { LiveToolPill } from '../../utils/liveActTrail';
import type { StepCall, StepWork } from '../../utils/turnSteps';
import { useSessionStore } from '../../stores/session';
import { usePermissionsStore, type PermissionRequest } from '../../stores/permissions';
import { actionLabel, waitedSeconds } from '../../utils/permissionAsk';
import ThinkingAndCalls from './ThinkingAndCalls.vue';

const props = withDefaults(
  defineProps<{
    /** The exchange's live act-trail pills. */
    pills?: LiveToolPill[];
    /** The stored row this step runs for, its live calls left out. */
    work?: StepWork | null;
    /** False where a stop interrupts without handing anything back to undo (a
     *  delegate's transcript). */
    undoable?: boolean;
    /** The permission ask this turn is parked on, if any. */
    ask?: PermissionRequest | null;
  }>(),
  { pills: () => [], work: null, undoable: true, ask: null },
);

const session = useSessionStore();
const permissions = usePermissionsStore();
const rootRef = ref<HTMLElement | null>(null);

const work = computed<StepWork>(() => ({
  thinking: props.work?.thinking,
  tool_calls: [
    ...(props.work?.tool_calls ?? []),
    ...props.pills.map((p): StepCall => ({
      tool_name: p.name,
      summary: p.summary ?? '',
      state: p.resolved ? (p.ok ? 'done' : 'error') : 'started',
      delegate: p.delegate,
    })),
  ],
}));

// The ask's "waiting" age ticks while it shows, and while this step shows it
// the input dock leaves it out of its off-screen list.
const now = ref(Date.now());
let timer: ReturnType<typeof setInterval> | undefined;

watch(
  () => props.ask?.request_id,
  (requestId, prevId) => {
    if (prevId) permissions.unmarkShown(prevId);
    if (requestId) permissions.markShown(requestId);
    clearInterval(timer);
    timer = requestId ? setInterval(() => (now.value = Date.now()), 100) : undefined;
  },
  { immediate: true },
);

onUnmounted(() => {
  clearInterval(timer);
  if (props.ask) permissions.unmarkShown(props.ask.request_id);
});

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
</script>

<template>
  <ThinkingAndCalls
    class="activity"
    :work="work"
    :label="ask && (ask.summary || actionLabel(ask.action_id))"
    live
  >
    <span ref="rootRef" class="activity__controls">
      <template v-if="ask">
        <span class="activity__wait">· waiting for you · {{ waitedSeconds(ask.asked_at, now) }}s</span>
        <span v-if="ask.summary" class="activity__ask">{{ actionLabel(ask.action_id) }}</span>
        <button
          type="button"
          class="activity__answer activity__answer--deny"
          @click="permissions.respond(ask.request_id, false)"
        >
          Deny
        </button>
        <button
          type="button"
          class="activity__answer activity__answer--allow"
          @click="permissions.respond(ask.request_id, true)"
        >
          Allow
        </button>
      </template>
      <button
        class="activity__stop"
        :aria-label="undoable ? 'Stop and undo' : 'Stop subagent'"
        :title="undoable ? 'Stop & undo' : 'Stop subagent'"
        type="button"
        @click="onStop"
      >Stop</button>
    </span>
  </ThinkingAndCalls>
</template>
