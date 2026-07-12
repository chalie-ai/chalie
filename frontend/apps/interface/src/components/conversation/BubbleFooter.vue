<script setup lang="ts">
import { computed, ref } from 'vue';
import type { ConversationMessage } from '../../api/conversation';
import { messagePlaintext } from '../../utils/speech';
import { emit as busEmit } from '../../composables/useEventBus';
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

const speakText = computed(() => messagePlaintext(props.message));

const rootRef = ref<HTMLElement | null>(null);
const expanded = ref(false);

// Speak plays the WHOLE turn (every assistant row, not just this exchange) —
// read straight off the rendered DOM (the `data-speech` attribute every
// Chalie bubble under this turn carries), matching the DOM-contract (no
// shared client store).
function onSpeak(): void {
  const turnHost = rootRef.value?.closest<HTMLElement>('[data-turn-id]');
  if (!turnHost) return;
  const text = Array.from(
    turnHost.querySelectorAll<HTMLElement>('.speech-form--chalie[data-speech]'),
  )
    .map((el) => el.dataset.speech ?? '')
    .filter(Boolean)
    .join(' ');
  if (text) busEmit('chalie:speak-message', { text });
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
          v-if="speakText"
          class="speech-form__act-btn speech-form__act-btn--speak"
          aria-label="Read this message aloud"
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
