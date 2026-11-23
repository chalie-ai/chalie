<script setup lang="ts">
/**
 * ActivityLine — the running step of a still-working exchange: the same work a
 * settled row shows (ThinkingAndCalls), live — the stored row's thinking and
 * calls once a refetch returned it, then the act-trail pills running now — its
 * head carrying the turn's controls. Stop always; while the turn is parked on
 * an ask, the head names the ask, a ticking "· waiting for you · Ns" note
 * follows — aged from the server's `asked_at` stamp, so it survives a reload —
 * and the ask is answered on the line itself: a permission on the head
 * (Deny/Allow), a sign-in code typed on the head, a question with its answers
 * under the head.
 */
import { computed, onUnmounted, ref, watch } from 'vue';
import { readDomContext } from '../../utils/domContext';
import { lastUserText } from '../../utils/turnDom';
import type { LiveToolPill } from '../../utils/liveActTrail';
import type { StepCall, StepWork } from '../../utils/turnSteps';
import { useSessionStore } from '../../stores/session';
import {
  AskCodeEntry,
  actionLabel,
  CODE_ASK_WORDS,
  QuestionControls,
  useAsksStore,
  waitedSeconds,
  type InlineAsk,
} from '@chalie/shared';
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
    /** The permission ask, question or sign-in code ask this turn is parked on, if any. */
    ask?: InlineAsk | null;
    /** The caller's subagents reply late (see `callMark`). */
    repliesLate?: boolean;
  }>(),
  { pills: () => [], work: null, undoable: true, ask: null, repliesLate: false },
);

const session = useSessionStore();
const asks = useAsksStore();
const rootRef = ref<HTMLElement | null>(null);

const askLabel = computed(() => {
  if (props.ask == null) return null;
  if (props.ask.kind === 'question') return 'Chalie has a question';
  if (props.ask.kind === 'code') return CODE_ASK_WORDS[props.ask.data.service].heading;
  return props.ask.data.summary || actionLabel(props.ask.data.action_id);
});

const work = computed<StepWork>(() => ({
  thinking: props.work?.thinking,
  tool_calls: [
    ...(props.work?.tool_calls ?? []),
    ...props.pills.map((p): StepCall => ({
      id: Number(p.id),
      tool_name: p.name,
      params: p.params,
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
  () => props.ask?.ask_id,
  (askId, prevId) => {
    if (prevId) asks.unmarkShown(prevId);
    if (askId) asks.markShown(askId);
    clearInterval(timer);
    timer = askId ? setInterval(() => (now.value = Date.now()), 100) : undefined;
  },
  { immediate: true },
);

onUnmounted(() => {
  clearInterval(timer);
  if (props.ask) asks.unmarkShown(props.ask.ask_id);
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
    :label="askLabel"
    :replies-late="repliesLate"
    live
  >
    <span ref="rootRef" class="activity__controls">
      <span v-if="ask" class="activity__wait">· waiting for you · {{ waitedSeconds(ask.asked_at, now) }}s</span>
      <template v-if="ask?.kind === 'permission'">
        <span v-if="ask.data.summary" class="activity__ask">{{ actionLabel(ask.data.action_id) }}</span>
        <button
          type="button"
          class="activity__answer activity__answer--deny"
          @click="asks.reply(ask.ask_id, false)"
        >
          Deny
        </button>
        <button
          type="button"
          class="activity__answer activity__answer--allow"
          @click="asks.reply(ask.ask_id, true)"
        >
          Allow
        </button>
      </template>
      <AskCodeEntry v-else-if="ask?.kind === 'code'" :ask="ask" />
      <button
        class="activity__stop"
        :aria-label="undoable ? 'Stop and undo' : 'Stop subagent'"
        :title="undoable ? 'Stop & undo' : 'Stop subagent'"
        type="button"
        @click="onStop"
      >Stop</button>
    </span>
    <QuestionControls v-if="ask?.kind === 'question'" :ask="ask" />
  </ThinkingAndCalls>
</template>
