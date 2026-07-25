<script setup lang="ts">
import { computed, ref } from 'vue';
import type { ConversationMessage } from '../../api/conversation';
import { messagePlaintext } from '../../utils/speech';
import { emit as busEmit } from '../../composables/useEventBus';
import { useVoiceTranscriptsStore } from '../../stores/voiceTranscripts';
import { voice } from '../../api/voice';
import { Copy, Reply, Volume2 } from '@lucide/vue';

const props = withDefaults(defineProps<{
  message: ConversationMessage;
  canReply?: boolean;
  toolCalls?: NonNullable<ConversationMessage['tool_calls']>;
  threadPill?: { status: 'working' | 'done' | 'thread' | 'idle'; label: string } | null;
}>(), {
  canReply: false,
  toolCalls: () => [],
  threadPill: null,
});

const emit = defineEmits<{ reply: []; openThread: [] }>();

const rootRef = ref<HTMLElement | null>(null);
const expanded = ref(false);

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
</script>

<template>
  <div ref="rootRef" class="speech-form__meta-wrap">
    <div class="speech-form__meta">
      <span class="speech-form__timestamp">{{ message.timestamp }}</span>

      <button
        v-if="toolCalls.length > 0"
        class="trace-pill"
        :class="{ 'trace-pill--open': expanded }"
        :aria-expanded="expanded"
        type="button"
        @click="expanded = !expanded"
      >
        <span class="trace-pill__dot" aria-hidden="true" />
        {{ toolCalls.length }} tool{{ toolCalls.length === 1 ? '' : 's' }} used
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

    <div
      v-if="toolCalls.length > 0"
      class="trace-body"
      :class="{ 'trace-body--open': expanded }"
    >
      <div class="trace-body__inner">
        <div class="calls">
          <div
            v-for="(c, i) in toolCalls"
            :key="i"
            class="call"
            :class="{ 'call--error': c.state === 'error' }"
          >
            <span class="call__fn">{{ c.tool_name }}</span>
            <span class="call__summary">{{ c.summary }}</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
