<script setup lang="ts">
import { computed, ref } from 'vue';
import type { ConversationMessage } from '../../api/conversation';
import { messagePlaintext } from '../../utils/speech';
import { emit as busEmit } from '../../composables/useEventBus';
import { Reply, Volume2 } from '@lucide/vue';

const props = withDefaults(defineProps<{ message: ConversationMessage; canReply?: boolean }>(), {
  canReply: false,
});

const emit = defineEmits<{ reply: [] }>();

const speakText = computed(() => messagePlaintext(props.message));

const rootRef = ref<HTMLElement | null>(null);

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
</script>

<template>
  <div ref="rootRef" class="speech-form__meta">
    <span class="speech-form__timestamp">{{ message.timestamp }}</span>

    <span class="speech-form__meta-spacer"></span>

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
      v-if="canReply"
      class="speech-form__act-btn speech-form__act-btn--reply"
      aria-label="Reply in a thread"
      type="button"
      @click="emit('reply')"
    >
      <Reply :size="16" />
    </button>
  </div>
</template>
