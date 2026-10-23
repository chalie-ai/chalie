<script setup lang="ts">
import { computed, ref } from 'vue';
import type { ConversationMessage, PlanTask } from '../../api/conversation';
import { messagePlaintext } from '../../utils/speech';
import { emit as busEmit } from '../../composables/useEventBus';
import { useVoiceTranscriptsStore } from '../../stores/voiceTranscripts';
import { voice } from '../../api/voice';
import { Copy, Reply, Volume2 } from '@lucide/vue';
import StepList from './StepList.vue';
import PlanCard from './PlanCard.vue';

const props = withDefaults(defineProps<{
  message: ConversationMessage;
  canReply?: boolean;
  threadPill?: { status: 'working' | 'done' | 'thread' | 'idle'; label: string } | null;
  /** The exchange's rows this answer closes — including the answer row itself —
      behind the work toggle ("N step(s)"). */
  work?: ConversationMessage[];
  /** The exchange's wall-clock duration in ms, read next to the step count. */
  durationMs?: number | null;
  /** The plan the exchange kept, behind the plan link ("plan N/M"). */
  plan?: PlanTask[];
}>(), {
  canReply: false,
  threadPill: null,
  work: () => [],
  durationMs: null,
  plan: () => [],
});

const emit = defineEmits<{ reply: []; openThread: [] }>();

const voiceStore = useVoiceTranscriptsStore();

// Only the row that CLOSES a turn is spoken, so only it gets a button — the
// mid-turn "let me check…" rows have no audio and never will.
const transcriptId = computed(() => Number(props.message.id));
const canSpeak = computed(() => !!props.message.settled && Number.isFinite(transcriptId.value));

// Live state wins over the snapshot this message was fetched with; null means
// nothing has been attempted, which is normal for history that predates
// pre-synthesis and reads as a plain, pressable button.
const voiceState = computed(() =>
  voiceStore.stateFor(transcriptId.value, props.message.voice_state ?? null),
);

// Pending fades slowly in and out — synthesis is seconds, not milliseconds,
// and a press that appears to do nothing reads as broken. Failed is terminal:
// the pipeline exhausted its attempts, so the button stays red and dead rather
// than inviting a press that can only fail again.
const speakDisabled = computed(() => voiceState.value === 'pending' || voiceState.value === 'failed');
const speakLabel = computed(() => {
  if (voiceState.value === 'pending') return 'Preparing speech…';
  if (voiceState.value === 'failed') return 'Speech could not be generated for this message';
  return 'Read this message aloud';
});

// A row with stored audio opens the player straight away. One with no attempt
// on record has to start the pipeline first — the GET does that and answers
// 202, and the pipeline's own WS frames drive the button from there.
async function onSpeak(): Promise<void> {
  if (speakDisabled.value) return;
  if (voiceState.value === 'ready') {
    busEmit('chalie:speak-message', { transcriptId: transcriptId.value });
    return;
  }
  voiceStore.record(transcriptId.value, 'pending');
  try {
    const resp = await voice.transcript(transcriptId.value);
    if (resp.ok) {
      voiceStore.record(transcriptId.value, 'ready');
      busEmit('chalie:speak-message', { transcriptId: transcriptId.value });
      return;
    }
    // 202 leaves it pending — the pipeline is running and will push its own
    // terminal frame. Anything else is terminal here and now.
    if (resp.status !== 202) voiceStore.record(transcriptId.value, 'failed');
  } catch (err) {
    console.error('[BubbleFooter] speech request failed:', err);
    voiceStore.record(transcriptId.value, 'failed');
  }
}

// Copy message text to clipboard — guard for absence, no throw.
function onCopy(): void {
  const text = messagePlaintext(props.message);
  if (!navigator.clipboard) return;
  navigator.clipboard.writeText(text).catch(() => {
    // Silently swallow — clipboard writes can fail in non-secure contexts.
  });
}

// ── Work toggle — the steps behind this answer, folded open below the footer ─
//
// Owner ruling: the work sits below the answer, in its footer. The label shows
// "N step(s)" (+ the exchange's duration) when the answer did any work, and
// "thought for Xs" whenever it reasoned — beside the steps label on a tool
// turn — each opening its own list; the step list itself never carries the
// reasoning. "plan N/M" beside them opens the plan the exchange kept. Nothing
// for a plain reply.

type Panel = 'steps' | 'thought' | 'plan';

const panel = ref<Panel | null>(null);

/** One fold, one open panel: open `which`'s list, close it when it is the
 *  open one, switch to it when the other is open. */
function togglePanel(which: Panel): void {
  panel.value = panel.value === which ? null : which;
}

const stepCount = computed(
  () =>
    props.work.reduce((n, m) => n + (m.tool_calls?.length ?? 0), 0) +
    props.work.filter((m) => m.role === 'assistant' && m.content.trim() !== '' && m.id !== props.message.id).length,
);
const failCount = computed(() =>
  props.work.reduce((n, m) => n + (m.tool_calls?.filter((c) => c.state === 'error').length ?? 0), 0),
);
const thinkMs = computed(() => props.work.reduce((n, m) => n + (m.thinking?.duration_ms ?? 0), 0));

// "Xs", "Xm Ys" or "Xh Ym" — rounded seconds, the same reading as the thinking link.
function formatDuration(ms: number): string {
  const seconds = Math.round(ms / 1000);
  const hours = Math.floor(seconds / 3600);
  const mins = Math.floor(seconds / 60) % 60;
  if (hours > 0) return `${hours}h ${mins}m`;
  return mins === 0 ? `${seconds}s` : `${mins}m ${seconds % 60}s`;
}

// "N step(s)" (+ the exchange's duration) — null when the answer did no work.
const stepsLabel = computed<string | null>(() => {
  if (stepCount.value === 0) return null;
  const n = stepCount.value;
  const base = `${n} step${n === 1 ? '' : 's'}`;
  return props.durationMs && props.durationMs > 0 ? `${base} · ${formatDuration(props.durationMs)}` : base;
});

// "thought for Xs" — null when the exchange never reasoned.
const thoughtLabel = computed<string | null>(
  () => (thinkMs.value > 0 ? `thought for ${formatDuration(thinkMs.value)}` : null),
);

// "plan N/M" — done tasks of all; null when the exchange kept no plan.
const planLabel = computed<string | null>(() =>
  props.plan.length > 0
    ? `plan ${props.plan.filter((t) => t.status === 'done').length}/${props.plan.length}`
    : null,
);

// Reasoning traces for the thought panel — a reply that only thought, or a
// tool turn that reasoned alongside its steps.
const thinkTraces = computed(() => props.work.flatMap((m) => m.thinking?.traces ?? []));
</script>

<template>
  <div class="speech-form__meta-wrap">
    <div class="speech-form__meta">
      <span class="speech-form__timestamp">{{ message.timestamp }}</span>

      <!-- The work behind this answer — "N step(s) · Xm Ys" (plus "thought
           for Xs" beside it whenever the exchange reasoned), each folding its
           own list open below the footer. -->
      <button
        v-if="stepsLabel"
        type="button"
        class="speech-form__work"
        :aria-expanded="panel === 'steps'"
        @click="togglePanel('steps')"
      >
        · {{ stepsLabel }}
        <span v-if="failCount" class="activity__fail"> · {{ failCount }} failed</span>
        <span class="activity__chev" aria-hidden="true">›</span>
      </button>
      <button
        v-if="thoughtLabel"
        type="button"
        class="speech-form__work"
        :aria-expanded="panel === 'thought'"
        @click="togglePanel('thought')"
      >
        · {{ thoughtLabel }}
        <span class="activity__chev" aria-hidden="true">›</span>
      </button>
      <button
        v-if="planLabel"
        type="button"
        class="speech-form__work"
        :aria-expanded="panel === 'plan'"
        @click="togglePanel('plan')"
      >
        · {{ planLabel }}
        <span class="activity__chev" aria-hidden="true">›</span>
      </button>

      <button
        v-if="threadPill"
        class="thread-pill"
        :class="`thread-pill--${threadPill.status}`"
        type="button"
        @click="emit('openThread')"
      >
        <span class="thread-pill__dot" aria-hidden="true" />
        <span class="thread-pill__label">{{ threadPill.label }}</span>
        <span class="thread-pill__chevron" aria-hidden="true">›</span>
      </button>

      <span class="speech-form__acts">
        <button
          v-if="canSpeak"
          class="speech-form__act-btn speech-form__act-btn--speak"
          :class="{
            'speech-form__act-btn--pending': voiceState === 'pending',
            'speech-form__act-btn--failed': voiceState === 'failed',
          }"
          :aria-label="speakLabel"
          :title="speakLabel"
          :disabled="speakDisabled"
          type="button"
          @click="onSpeak"
        >
          <Volume2 :size="16" />
        </button>

        <button
          class="speech-form__act-btn speech-form__act-btn--copy"
          aria-label="Copy message"
          type="button"
          @click="onCopy"
        >
          <Copy :size="16" />
        </button>

        <button
          v-if="canReply && !threadPill"
          class="speech-form__act-btn speech-form__act-btn--reply"
          aria-label="Reply in a thread"
          type="button"
          @click="emit('reply')"
        >
          <Reply :size="16" />
        </button>
      </span>
    </div>

    <!-- inert while folded: a delegate call row is a button, and the fold only
         collapses its height, so it would stay reachable by Tab unseen. The
         open panel names the content — the plan under plan, the step list
         under steps (it stays in the DOM while folded, height-collapsed), the
         reasoning traces under thought. A reply that only thought has nothing
         but traces. -->
    <div
      v-if="stepsLabel || thoughtLabel || planLabel"
      class="trace-body"
      :class="{ 'trace-body--open': panel !== null }"
      :inert="panel === null"
    >
      <div class="trace-body__inner">
        <PlanCard v-if="panel === 'plan'" :tasks="plan" />
        <StepList
          v-else-if="stepCount && panel !== 'thought'"
          :messages="work"
          :answer-id="message.id"
        />
        <div v-else class="activity__steps">
          <pre
            v-for="(trace, j) in thinkTraces"
            :key="j"
            class="activity__trace"
          >{{ trace }}</pre>
        </div>
      </div>
    </div>
  </div>
</template>
