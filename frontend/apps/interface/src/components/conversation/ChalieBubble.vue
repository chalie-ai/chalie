<script setup lang="ts">
import { computed, ref } from 'vue';
import type { ConversationMessage } from '../../api/conversation';
import { messagePlaintext } from '../../utils/speech';
import { renderMarkup } from '../../composables/useMarkup';
import { emit as busEmit } from '../../composables/useEventBus';
import { Reply, Volume2 } from '@lucide/vue';
import SegmentRenderer from './SegmentRenderer.vue';

const props = withDefaults(
  defineProps<{ message: ConversationMessage; isLast?: boolean; canReply?: boolean }>(),
  { isLast: false, canReply: false },
);

const emit = defineEmits<{ reply: [] }>();

const speakText = computed(() => messagePlaintext(props.message));

const rootRef = ref<HTMLElement | null>(null);

// Speak plays the WHOLE turn (every assistant row), not just this row) — read
// straight off the rendered DOM (the `data-speech` attribute every Chalie
// bubble in the same turn carries) rather than the retired buffer's
// `turnSpeechText`, matching the DOM-contract (no shared client store).
function onSpeak(): void {
  const turnHost = rootRef.value?.closest<HTMLElement>('[data-turn-id]');
  if (!turnHost) return;
  const text = Array.from(turnHost.querySelectorAll<HTMLElement>('.speech-form--chalie[data-speech]'))
    .map((el) => el.dataset.speech ?? '')
    .filter(Boolean)
    .join(' ');
  if (text) busEmit('chalie:speak-message', { text });
}
</script>

<template>
  <div
    ref="rootRef"
    class="speech-form speech-form--chalie"
    :data-transcript-row-id="message.id"
    :data-speech="speakText"
  >
    <SegmentRenderer
      v-if="message.segments && message.segments.length"
      :segments="message.segments"
    />
    <div
      v-else
      class="speech-form__text chalie-markup"
      v-html="renderMarkup(message.content ?? '')"
    />

    <!-- Footer lives only on the turn's LAST assistant row. -->
    <div v-if="isLast" class="speech-form__meta">
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
  </div>
</template>

<style lang="scss">
/*
 * Styles targeting v-html'd inner content (chalie-markup, chalie-action-button,
 * chalie-code, auto-linkified <a>) MUST be global — Vue scoped CSS does not
 * reach nodes injected by v-html. Structural bubble layout lives in conversation.scss.
 */
.chalie-markup {
  word-break: break-word;
  position: relative;
  z-index: 2;
}

.chalie-markup p {
  margin: 0 0 0.75em;
  line-height: 1.35;
}

.chalie-markup p:last-child {
  margin-bottom: 0;
}

.chalie-markup a {
  color: var(--accent-primary);
  text-decoration: underline;
  text-underline-offset: 2px;
}

.chalie-markup a:hover {
  color: var(--accent-secondary);
}

.chalie-code,
.chalie-markup code {
  font-family: var(--font-mono);
  font-size: 0.875em;
  background: color-mix(in oklab, var(--text-primary) 8%, transparent);
  padding: 0.1em 0.35em;
  border-radius: 4px;
}

.chalie-markup pre {
  background: color-mix(in oklab, var(--text-primary) 6%, transparent);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: var(--space-md);
  overflow-x: auto;
  margin: 0.75em 0;
}

.chalie-markup pre code {
  background: none;
  padding: 0;
  border-radius: 0;
  font-size: 0.85em;
}

.chalie-markup strong {
  font-weight: 600;
  color: var(--text-primary);
}

.chalie-markup em {
  font-style: italic;
  color: var(--text-secondary);
}

.chalie-markup ul,
.chalie-markup ol {
  padding-left: 1.5em;
  margin: 0.5em 0;
}

.chalie-markup ul {
  list-style: disc;
}

.chalie-markup ol {
  list-style: decimal;
}

.chalie-markup li {
  margin: 0.2em 0;
  line-height: 1.35;
}

.chalie-markup blockquote {
  border-left: 2px solid var(--border);
  margin: 0.75em 0;
  padding: 0.25em 0 0.25em var(--space-md);
  color: var(--text-secondary);
}

.chalie-markup hr {
  border: none;
  border-top: 1px solid var(--border);
  margin: var(--space-md) 0;
}

.chalie-markup table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.875em;
}

.chalie-markup th,
.chalie-markup td {
  border: 1px solid var(--border);
  padding: 6px 10px;
  text-align: left;
}

.chalie-markup thead {
  background: color-mix(in oklab, var(--text-primary) 5%, transparent);
}

.chalie-markup tbody tr:nth-child(even) {
  background: color-mix(in oklab, var(--text-primary) 3%, transparent);
}

/* Action buttons (from <actions>/<action> custom elements wired by useMarkup) */
.chalie-actions-row {
  display: flex;
  gap: var(--space-sm);
  margin-top: var(--space-md);
  flex-wrap: wrap;
}

.chalie-action-button {
  padding: 6px 16px;
  border-radius: var(--radius-full);
  border: 1px solid color-mix(in oklab, var(--violet) 35%, transparent);
  background: color-mix(in oklab, var(--violet) 8%, transparent);
  color: var(--accent);
  font-size: 0.82rem;
  font-weight: 500;
  letter-spacing: 0.02em;
  cursor: pointer;
  transition: all 220ms ease;
}

.chalie-action-button:hover:not([disabled]) {
  background: color-mix(in oklab, var(--violet) 18%, transparent);
  border-color: color-mix(in oklab, var(--violet) 55%, transparent);
}

.chalie-action-button--secondary {
  border-color: var(--border);
  background: transparent;
  color: var(--text-secondary);
}

.chalie-action-button--danger {
  border-color: color-mix(in oklab, var(--error) 40%, transparent);
  background: color-mix(in oklab, var(--error) 8%, transparent);
  color: var(--error);
}

.chalie-action-button--selected {
  background: color-mix(in oklab, var(--violet) 22%, transparent);
  border-color: var(--accent);
  color: var(--text-primary);
}

.chalie-action-button[disabled] {
  opacity: 0.4;
  cursor: default;
  pointer-events: none;
}
</style>
