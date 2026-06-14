<script setup lang="ts">
import type { ChalieForm } from '../../stores/conversation';
import { renderMarkup, extractText } from '../../composables/useMarkup';
import { emit } from '../../composables/useEventBus';
import SegmentRenderer from './SegmentRenderer.vue';

const props = defineProps<{ form: ChalieForm }>();

function onRemember(): void {
  // 150ms micro-delay mirrors legacy renderer.js
  emit('chalie:pin-moment', { content: props.form.text ?? '' });
}

function onSpeak(): void {
  const text = extractText(props.form.text ?? '');
  emit('chalie:speak-message', { text });
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
    <!-- Header row: glyph + timestamp -->
    <div class="speech-form__header">
      <svg class="sender-glyph" viewBox="0 0 18 18" fill="none" aria-hidden="true">
        <line x1="4" y1="14" x2="9" y2="5" stroke="currentColor" stroke-width="0.6" opacity="0.45" />
        <line x1="9" y1="5" x2="14" y2="11" stroke="currentColor" stroke-width="0.6" opacity="0.45" />
        <line x1="4" y1="14" x2="14" y2="11" stroke="currentColor" stroke-width="0.6" opacity="0.45" />
        <circle class="sender-glyph__dot" cx="4" cy="14" r="1.4" fill="currentColor" />
        <circle class="sender-glyph__dot" cx="9" cy="5" r="1.6" fill="currentColor" />
        <circle class="sender-glyph__dot" cx="14" cy="11" r="1.3" fill="currentColor" />
        <circle class="sender-glyph__dot" cx="11" cy="2" r="0.9" fill="currentColor" opacity="0.65" />
      </svg>
      <span class="speech-form__timestamp">{{ form.meta.ts ?? '' }}</span>
    </div>

    <!-- Content: segments path or single text path -->
    <SegmentRenderer
      v-if="form.meta.segments && form.meta.segments.length"
      :segments="form.meta.segments"
    />
    <div
      v-else
      class="speech-form__text chalie-markup"
      v-html="renderMarkup(form.text ?? '')"
    />

    <!-- Meta row: remember + speak buttons -->
    <div class="speech-form__meta">
      <button
        class="speech-form__remember-btn"
        aria-label="Remember this"
        type="button"
        @click="onRemember"
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
          stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M12 2l2.09 6.26L20 10l-5.91 1.74L12 18l-2.09-6.26L4 10l5.91-1.74L12 2z" />
        </svg>
      </button>
      <button
        class="speech-form__speak-btn"
        aria-label="Listen to this message"
        type="button"
        @click="onSpeak"
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
          stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
          <path d="M15.54 8.46a5 5 0 0 1 0 7.07" />
          <path d="M19.07 4.93a10 10 0 0 1 0 14.14" />
        </svg>
      </button>
    </div>
  </div>
</template>

<style>
/*
 * Styles targeting v-html'd inner content (chalie-markup, chalie-action-button,
 * chalie-code, auto-linkified <a>) MUST be global — Vue scoped CSS does not
 * reach nodes injected by v-html. Structural bubble layout remains scoped.
 *
 * These mirror the relevant rules from the legacy style.css.
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
  border: 1px solid rgba(138, 92, 255, 0.35);
  background: rgba(138, 92, 255, 0.08);
  color: var(--accent);
  font-size: 0.82rem;
  font-weight: 500;
  letter-spacing: 0.02em;
  cursor: pointer;
  transition: all 220ms ease;
}

.chalie-action-button:hover:not([disabled]) {
  background: rgba(138, 92, 255, 0.18);
  border-color: rgba(138, 92, 255, 0.55);
}

.chalie-action-button--secondary {
  border-color: var(--border);
  background: transparent;
  color: var(--text-secondary);
}

.chalie-action-button--danger {
  border-color: rgba(224, 108, 108, 0.4);
  background: rgba(224, 108, 108, 0.08);
  color: var(--error);
}

.chalie-action-button--selected {
  background: rgba(138, 92, 255, 0.22);
  border-color: var(--accent);
  color: var(--text-primary);
}

.chalie-action-button[disabled] {
  opacity: 0.4;
  cursor: default;
  pointer-events: none;
}
</style>
