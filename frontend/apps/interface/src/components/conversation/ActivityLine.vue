<script setup lang="ts">
/**
 * ActivityLine — one quiet line per assistant segment, above the answer.
 *
 * It folds every intermediate row of the segment (thinking traces, mid-turn
 * prose, tool calls) into an expandable step list, so a turn reads as ONE
 * activity line plus its answer instead of one bubble + "N tools used" pill
 * per LLM call. Live, the same line tracks the current act-trail pill (the
 * last unresolved one) with a ticking elapsed time and a Stop button: stop via
 * readDomContext/lastUserText/session.requestStop, a 100 ms clock that ticks
 * only while a pill is unresolved, and the undoable/subagent aria-labels.
 */
import { computed, onUnmounted, ref, watch } from 'vue';
import { Bot } from '@lucide/vue';
import type { ConversationMessage } from '../../api/conversation';
import { readDomContext } from '../../utils/domContext';
import { lastUserText } from '../../utils/turnDom';
import type { LiveToolPill } from '../../utils/liveActTrail';
import { useSessionStore } from '../../stores/session';
import { delegatePillAttrs } from '../../composables/useDelegatePill';
import ChalieBubble from './ChalieBubble.vue';
import ToolCallList from './ToolCallList.vue';

const props = withDefaults(
  defineProps<{
    /** The segment's rows, in order — the work this line summarises. */
    messages: ConversationMessage[];
    /** The answer row's id — excluded from the step list (it renders below). */
    answerId: string | null;
    /** True while the segment is working — live header + pills. */
    live?: boolean;
    /** The segment's live act-trail pills. */
    pills?: LiveToolPill[];
    /** False where a stop interrupts without handing anything back to undo (a
     *  delegate's transcript). */
    undoable?: boolean;
  }>(),
  { live: false, pills: () => [], undoable: true },
);

const session = useSessionStore();
const rootRef = ref<HTMLElement | null>(null);
const expanded = ref(false);

// ── Steps — the segment's work, in row order ─────────────────────────────────
//
// Each row contributes, in this order: its thinking (a 'thought' step), its
// prose (a 'prose' step — unless it IS the answer row), its tool calls (a
// 'calls' step). The answer row never becomes a prose step: it renders below
// the line, not inside it.

type ActivityStep =
  | { kind: 'thought'; traces: string[]; durationMs: number }
  | { kind: 'prose'; message: ConversationMessage }
  | { kind: 'calls'; calls: NonNullable<ConversationMessage['tool_calls']> };

const steps = computed<ActivityStep[]>(() => {
  const out: ActivityStep[] = [];
  for (const m of props.messages) {
    if (m.thinking) {
      out.push({ kind: 'thought', traces: m.thinking.traces, durationMs: m.thinking.duration_ms });
    }
    if (m.role === 'assistant' && m.content.trim() !== '' && String(m.id) !== props.answerId) {
      out.push({ kind: 'prose', message: m });
    }
    if (m.tool_calls?.length) {
      out.push({ kind: 'calls', calls: m.tool_calls });
    }
  }
  return out;
});

// ── Header numbers ────────────────────────────────────────────────────────────

const callCount = computed(() => props.messages.reduce((n, m) => n + (m.tool_calls?.length ?? 0), 0));
const failCount = computed(() =>
  props.messages.reduce((n, m) => n + (m.tool_calls?.filter((c) => c.state === 'error').length ?? 0), 0),
);
const thinkMs = computed(() => props.messages.reduce((n, m) => n + (m.thinking?.duration_ms ?? 0), 0));
const proseCount = computed(() => steps.value.filter((s) => s.kind === 'prose').length);

// "Xs" or "Xm Ys" — rounded seconds, the same reading as the thinking link.
function formatDuration(ms: number): string {
  const seconds = Math.round(ms / 1000);
  const mins = Math.floor(seconds / 60);
  const rem = seconds % 60;
  return mins === 0 ? `${seconds}s` : `${mins}m ${rem}s`;
}

// ── Live state: the last unresolved pill + a ticking clock ──────────────────

const lastUnresolvedPill = computed<LiveToolPill | null>(() => {
  for (let i = props.pills.length - 1; i >= 0; i--) {
    if (!props.pills[i].resolved) return props.pills[i];
  }
  return null;
});

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

// Live label: the last unresolved pill's summary (or name) + its whole
// elapsed seconds; before the first pill lands, the bare "Thinking…".
const liveLabel = computed(() => {
  const pill = lastUnresolvedPill.value;
  if (!pill) return 'Thinking…';
  const base = pill.summary || pill.name;
  const seconds = Math.max(0, Math.floor((now.value - pill.startedAt) / 1000));
  return `${base} · ${seconds}s`;
});

// Settled label: "N step(s)" (+ " · Xs" when any row thought), or a bare
// "Thought for Xs" when the segment is thinking with no tools or prose.
const settledLabel = computed(() => {
  const n = callCount.value + proseCount.value;
  if (n > 0) {
    const base = `${n} step${n === 1 ? '' : 's'}`;
    return thinkMs.value > 0 ? `${base} · ${formatDuration(thinkMs.value)}` : base;
  }
  return thinkMs.value > 0 ? `Thought for ${formatDuration(thinkMs.value)}` : '';
});

const headerLabel = computed(() => (props.live ? liveLabel.value : settledLabel.value));

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
  <!-- Nothing when the segment has no work and is not working. -->
  <div
    v-if="live || steps.length > 0"
    ref="rootRef"
    class="activity"
    :class="{ 'activity--live': live }"
  >
    <button
      class="activity__toggle"
      :aria-expanded="expanded"
      type="button"
      @click="expanded = !expanded"
    >
      <span class="activity__mark" />
      <span class="activity__label">{{ headerLabel }}</span>
      <span v-if="!live && failCount > 0" class="activity__fail">· {{ failCount }} failed</span>
      <span class="activity__chev" aria-hidden="true">›</span>
    </button>

    <button
      v-if="live"
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
        <div class="activity__steps">
          <template v-for="(step, i) in steps" :key="i">
            <details v-if="step.kind === 'thought'" class="activity__thought">
              <summary>Thought for {{ formatDuration(step.durationMs) }}</summary>
              <pre v-for="(trace, j) in step.traces" :key="j">{{ trace }}</pre>
            </details>
            <ChalieBubble
              v-else-if="step.kind === 'prose'"
              :message="step.message"
              class="activity__prose"
            />
            <ToolCallList v-else :calls="step.calls" />
          </template>

          <!-- Live act-trail pills, one row per tool call still in this turn. -->
          <template v-if="live">
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
                <span class="activity__pill-name">
                  <Bot v-if="pill.delegate" class="delegate-pill__icon" :size="14" aria-hidden="true" />{{
                    pill.name
                  }}
                </span>
                <span v-if="pill.summary" class="activity__pill-summary">— {{ pill.summary }}</span>
              </span>

              <span class="activity__pill-status">
                <template v-if="!pill.resolved">
                  <span class="activity__pill-elapsed">{{ pillSeconds(pill) }}s</span>
                </template>
                <template v-else-if="pill.ok">{{ pillSeconds(pill) }}s</template>
                <template v-else>error</template>
              </span>
            </component>
          </template>
        </div>
      </div>
    </div>
  </div>
</template>
