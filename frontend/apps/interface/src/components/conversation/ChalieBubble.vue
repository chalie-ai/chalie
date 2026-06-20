<script setup lang="ts">
import { ref, computed } from 'vue';
import type { ChalieForm } from '../../stores/conversation';
import { chalieFormPlaintext, useConversationStore } from '../../stores/conversation';
import { renderMarkup } from '../../composables/useMarkup';
import { emit } from '../../composables/useEventBus';
import { Star, Volume2 } from '@lucide/vue';
import SegmentRenderer from './SegmentRenderer.vue';

const conversationStore = useConversationStore();

const props = defineProps<{ form: ChalieForm }>();

// Prevents double-fire of the remember click.
const pinned = ref(false);
// Glow lands AFTER the 150ms delay (button disables immediately, --active later).
const pinActive = ref(false);

const MODE_LABELS: Record<string, string> = {
  ACT: 'acting',
  CLARIFY: 'clarifying',
  ACKNOWLEDGE: 'noting',
};

const modeBadgeLabel = computed(() => MODE_LABELS[props.form.meta.mode ?? ''] ?? '');

const speakText = computed(() => chalieFormPlaintext(props.form));

// Remember/speak controls live only on the turn's LAST Chalie row — a turn may
// span several assistant rows, and the controls act on the whole turn.
const isLastInTurn = computed(() => conversationStore.isLastChalieInTurn(props.form.id));

function onRemember(): void {
  if (pinned.value) return;
  pinned.value = true;
  setTimeout(() => {
    emit('chalie:pin-moment', { content: conversationStore.turnSpeechText(props.form.id) });
    pinActive.value = true;
  }, 150);
}

// Speak plays the WHOLE turn (every Chalie row), not just this row.
function onSpeak(): void {
  emit('chalie:speak-message', { text: conversationStore.turnSpeechText(props.form.id) });
}
</script>

<template>
  <div
    class="speech-form speech-form--chalie"
    :class="{
      'speech-form--escalation': form.escalation,
      'message--faded': form.inWorkingMemory === false,
    }"
  >
    <SegmentRenderer
      v-if="form.meta.segments && form.meta.segments.length"
      :segments="form.meta.segments"
    />
    <div
      v-else
      class="speech-form__text chalie-markup"
      v-html="renderMarkup(form.text ?? '')"
    />

    <!-- Footer lives only on the turn's LAST Chalie row; interim rows carry no meta. -->
    <div v-if="isLastInTurn" class="speech-form__meta">
      <span class="sender-glyph" aria-hidden="true"></span>
      <span class="speech-form__timestamp">{{ form.meta.ts ?? '' }}</span>

      <span v-if="modeBadgeLabel" class="meta-mode-badge">{{ modeBadgeLabel }}</span>

      <div class="speech-form__actions">
        <button
          class="speech-form__remember-btn"
          :class="{ 'speech-form__remember-btn--active': pinActive }"
          aria-label="Remember this"
          type="button"
          :disabled="pinned"
          @click="onRemember"
        >
          <Star :size="14" />
        </button>

        <button
          v-if="speakText"
          class="speech-form__speak-btn"
          aria-label="Listen to this message"
          type="button"
          @click="onSpeak"
        >
          <Volume2 :size="14" />
        </button>
      </div>
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
  line-height: 1.6;
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
  font-family: 'JetBrains Mono', 'Fira Code', ui-monospace, 'SF Mono', Menlo, monospace;
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
  line-height: 1.6;
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
