<script setup lang="ts">
import type { ConversationMessage } from '../../api/conversation';
import { renderMarkup } from '../../composables/useMarkup';
import SegmentRenderer from './SegmentRenderer.vue';

defineProps<{ message: ConversationMessage }>();
</script>

<template>
  <div
    class="speech-form speech-form--chalie"
    :data-transcript-row-id="message.id"
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
  </div>
</template>

<style lang="scss">
/*
 * Styles targeting v-html'd inner content (chalie-markup, chalie-code,
 * auto-linkified <a>) MUST be global — Vue scoped CSS does not
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
  color: var(--pink-text);
  text-decoration: underline;
  text-underline-offset: 2px;
}

.chalie-markup a:hover {
  color: var(--text);
}

.chalie-code,
.chalie-markup code {
  font-family: var(--font-mono);
  font-size: 0.875em;
  background: var(--surface-2);
  padding: 0.1em 0.35em;
}

.chalie-markup pre {
  background: var(--surface-2);
  border: 1px solid var(--line);
  padding: var(--space-md);
  overflow-x: auto;
  margin: 0.75em 0;
}

.chalie-markup pre code {
  background: none;
  padding: 0;
  font-size: 0.85em;
}

.chalie-markup strong {
  font-weight: 600;
  color: var(--text);
}

.chalie-markup em {
  font-style: italic;
  color: var(--muted);
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
  border-left: 2px solid var(--line);
  margin: 0.75em 0;
  padding: 0.25em 0 0.25em var(--space-md);
  color: var(--muted);
}

.chalie-markup hr {
  border: none;
  border-top: 1px solid var(--line);
  margin: var(--space-md) 0;
}

.chalie-markup table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.875em;
}

.chalie-markup th,
.chalie-markup td {
  border: 1px solid var(--line);
  padding: 6px 10px;
  text-align: left;
}

.chalie-markup thead,
.chalie-markup tbody tr:nth-child(even) {
  background: var(--surface-2);
}
</style>
