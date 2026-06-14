<script setup lang="ts">
import type { ConversationSegment } from '../../api/conversation';
import { renderMarkup } from '../../composables/useMarkup';
import { resolveRichCard } from '../rich/richRegistry';

defineProps<{ segments: ConversationSegment[] }>();
</script>

<template>
  <template v-for="(seg, idx) in segments" :key="idx">
    <!-- Text segment: render markup via v-html -->
    <div
      v-if="seg.type === 'text'"
      class="speech-form__text"
      v-html="renderMarkup(seg.content ?? '')"
    />

    <!-- Rich segment: resolve card component or fall back to synthesis/content text -->
    <template v-else-if="seg.type === 'rich'">
      <component
        :is="resolveRichCard(seg.tag ?? '')!.component"
        v-if="resolveRichCard(seg.tag ?? '')"
        :payload="seg.payload"
        :synthesis="seg.synthesis"
      />
      <!-- Unknown-tag fallback (primary P1a path): render synthesis or content as markup -->
      <div
        v-else
        class="speech-form__text"
        v-html="renderMarkup(seg.synthesis ?? seg.content ?? '')"
      />
    </template>
  </template>
</template>
