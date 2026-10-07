<script setup lang="ts">
/**
 * ActivityLine — the one live line above a still-working exchange.
 *
 * It renders only while the turn works: the exchange's in-progress work
 * (mid-turn prose and tool calls, via StepList) folded into an expandable
 * step list, the current act-trail pills with a ticking elapsed time, and a
 * Stop button (stop via readDomContext/lastUserText/session.requestStop, a
 * 100 ms clock that ticks while a pill is unresolved or the line is parked
 * on a permission ask). When the turn is parked on a gate, the label names
 * the ask and a ticking "· waiting for you · Ns" note follows it — aged from
 * the server's `asked_at` stamp, so it survives a reload. When the exchange
 * settles its work moves into the answer's footer (BubbleFooter) below, so
 * no line stands above a settled answer.
 */
import { computed, onUnmounted, ref, watch } from 'vue';
import { Bot } from '@lucide/vue';
import type { ConversationMessage } from '../../api/conversation';
import { readDomContext } from '../../utils/domContext';
import { lastUserText } from '../../utils/turnDom';
import type { LiveToolPill } from '../../utils/liveActTrail';
import { useSessionStore } from '../../stores/session';
import type { PermissionRequest } from '../../stores/permissions';
import { delegatePillAttrs } from '../../composables/useDelegatePill';
import StepList from './StepList.vue';

const props = withDefaults(
  defineProps<{
    /** The exchange's rows, in order — the work this line summarises. */
    messages: ConversationMessage[];
    /** The exchange's live act-trail pills. */
    pills?: LiveToolPill[];
    /** False where a stop interrupts without handing anything back to undo (a
     *  delegate's transcript). */
    undoable?: boolean;
    /** The permission ask this turn is parked on, if any — the label names it
     *  and a ticking "· waiting for you" note follows. */
    ask?: PermissionRequest | null;
  }>(),
  { pills: () => [], undoable: true, ask: null },
);

const session = useSessionStore();
const rootRef = ref<HTMLElement | null>(null);
const expanded = ref(false);

// ── Live state: the last unresolved pill + a ticking clock ──────────────────

const lastUnresolvedPill = computed<LiveToolPill | null>(() => {
  for (let i = props.pills.length - 1; i >= 0; i--) {
    if (!props.pills[i].resolved) return props.pills[i];
  }
  return null;
});

// Live timer: ticks while anything on the line is counting — a pill still
// unresolved, or the ask this turn is parked on (its "waiting" age ticks).
const now = ref(Date.now());
let timer: ReturnType<typeof setInterval> | null = null;

const hasRunning = computed(() => props.pills.some((p) => !p.resolved));
const clockLive = computed(() => hasRunning.value || props.ask != null);

function stopClock(): void {
  if (timer) clearInterval(timer);
  timer = null;
}

watch(
  clockLive,
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

// Live label: the parked ask's summary (or action id) while the turn waits
// on a gate; otherwise the last unresolved pill's summary (or name) + its
// whole elapsed seconds; before the first pill lands, the bare "Thinking…".
const liveLabel = computed(() => {
  if (props.ask) return props.ask.summary || props.ask.action_id;
  const pill = lastUnresolvedPill.value;
  if (!pill) return 'Thinking…';
  const base = pill.summary || pill.name;
  const seconds = Math.max(0, Math.floor((now.value - pill.startedAt) / 1000));
  return `${base} · ${seconds}s`;
});

// Whole seconds since the ask parked — 0 when the stamp is unparseable or in
// the future (a clock skew); the server stamp is what makes it survive a
// reload.
const askSeconds = computed(() => {
  if (!props.ask) return 0;
  const parked = Date.parse(props.ask.asked_at);
  return Number.isNaN(parked) ? 0 : Math.max(0, Math.floor((now.value - parked) / 1000));
});

// ── Stop ─────────────────────────────────────────────────────────────────────

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
  <div ref="rootRef" class="activity">
    <button
      class="activity__toggle"
      :aria-expanded="expanded"
      type="button"
      @click="expanded = !expanded"
    >
      <span class="activity__mark" />
      <span class="activity__label">{{ liveLabel }}</span>
      <!-- Inherits the toggle's muted colour — only .activity__label is
           text-coloured, so no style of its own. -->
      <span v-if="ask" class="activity__wait">· waiting for you · {{ askSeconds }}s</span>
      <span class="activity__chev" aria-hidden="true">›</span>
    </button>

    <button
      class="activity__stop"
      :aria-label="undoable ? 'Stop and undo' : 'Stop subagent'"
      :title="undoable ? 'Stop & undo' : 'Stop subagent'"
      type="button"
      @click="onStop"
    >Stop</button>

    <!-- inert while folded: a delegate call row is a button, and the fold only
         collapses its height, so it would stay reachable by Tab unseen. -->
    <div
      class="trace-body"
      :class="{ 'trace-body--open': expanded }"
      :inert="!expanded"
    >
      <div class="trace-body__inner">
        <StepList :messages="messages" :answer-id="null" />

        <!-- Live act-trail pills, one row per tool call still in this turn. -->
        <div class="activity__steps">
          <component
            :is="pill.delegate ? 'button' : 'div'"
            v-for="pill in pills"
            :key="pill.id"
            v-bind="delegatePillAttrs(pill.delegate)"
            class="activity__pill"
            :class="{
              'activity__pill--running': !pill.resolved,
              'activity__pill--done': pill.resolved && pill.ok,
              'activity__pill--error': pill.resolved && !pill.ok,
            }"
            :data-call-id="pill.id"
            :data-transcript-row-id="pill.transcriptRowId"
          >
            <span class="activity__pill-label">
              <span class="activity__pill-summary">
                <Bot v-if="pill.delegate" class="delegate-pill__icon" :size="16" aria-hidden="true" />{{
                  pill.summary || pill.name
                }}
              </span>
            </span>

            <span class="activity__pill-status">
              <template v-if="!pill.resolved">
                <span class="activity__pill-elapsed">{{ pillSeconds(pill) }}s</span>
              </template>
              <template v-else-if="pill.ok">{{ pillSeconds(pill) }}s</template>
              <template v-else>error</template>
            </span>
          </component>
        </div>
      </div>
    </div>
  </div>
</template>
