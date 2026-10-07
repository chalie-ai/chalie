<script setup lang="ts">
import { computed, ref } from 'vue';
import type { ConversationMessage } from '../../api/conversation';
import { messagePlaintext } from '../../utils/speech';
import { emit as busEmit } from '../../composables/useEventBus';
import { useVoiceTranscriptsStore } from '../../stores/voiceTranscripts';
import { voice } from '../../api/voice';
import { Copy, Reply, Volume2 } from '@lucide/vue';
import StepList from './StepList.vue';

const props = withDefaults(defineProps<{
  message: ConversationMessage;
  canReply?: boolean;
  threadPill?: { status: 'working' | 'done' | 'thread' | 'idle'; label: string } | null;
  /** The exchange's rows this answer closes — including the answer row itself —
      behind the work toggle ("N step(s)"). */
  work?: ConversationMessage[];
  /** The exchange's wall-clock duration in ms, read next to the step count. */
  durationMs?: number | null;
}>(), {
  canReply: false,
  threadPill: null,
  work: () => [],
  durationMs: null,
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
// "N step(s)" (+ the exchange's duration) when the answer did any work,
// "thought for Xs" when it only thought, and nothing for a plain reply.

const open = ref(false);

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

const label = computed<string | null>(() => {
  if (stepCount.value > 0) {
    const n = stepCount.value;
    const base = `${n} step${n === 1 ? '' : 's'}`;
    return props.durationMs && props.durationMs > 0 ? `${base} · ${formatDuration(props.durationMs)}` : base;
  }
  return thinkMs.value > 0 ? `thought for ${formatDuration(thinkMs.value)}` : null;
});

// Reasoning traces for the no-work fallback list (a reply that only thought).
const thinkTraces = computed(() => props.work.flatMap((m) => m.thinking?.traces ?? []));
</script>

<template>
  <div class="speech-form__meta-wrap">
    <div class="speech-form__meta">
      <span class="speech-form__timestamp">{{ message.timestamp }}</span>

      <!-- The work behind this answer — "N step(s) · Xm Ys" (or "thought for
           Xs"), folding its step list open below the footer. -->
      <button
        v-if="label"
        type="button"
        class="speech-form__work"
        :aria-expanded="open"
        @click="open = !open"
      >
        · {{ label }}
        <span v-if="failCount" class="activity__fail"> · {{ failCount }} failed</span>
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
         collapses its height, so it would stay reachable by Tab unseen. With no
         steps at all, a plain reply that only thought shows its reasoning
         traces instead. -->
    <div
      v-if="label"
      class="trace-body"
      :class="{ 'trace-body--open': open }"
      :inert="!open"
    >
      <div class="trace-body__inner">
        <StepList
          v-if="stepCount"
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
