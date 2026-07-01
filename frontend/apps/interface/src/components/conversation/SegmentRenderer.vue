<script setup lang="ts">
import { computed } from 'vue';
import type { ConversationSegment } from '../../api/conversation';
import { renderMarkup } from '../../composables/useMarkup';
import type { RichCardEntry } from '../rich/richRegistry';
import { resolveRichCard } from '../rich/richRegistry';

const props = defineProps<{ segments: ConversationSegment[] }>();

// Precompute resolved entries ONCE per segment so the template never calls
// resolveRichCard() twice and the non-null assertion is eliminated.
const resolved = computed<{ seg: ConversationSegment; richEntry: RichCardEntry | null }[]>(() =>
  props.segments.map((seg) => ({
    seg,
    richEntry: seg.type === 'rich' ? (resolveRichCard(seg.tag ?? '') ?? null) : null,
  })),
);
</script>

<template>
  <template v-for="(item, idx) in resolved" :key="idx">
    <div
      v-if="item.seg.type === 'text'"
      class="speech-form__text"
      v-html="renderMarkup(item.seg.content ?? '')"
    />

    <!-- Guard on a wrapping <template v-if> so vue-tsc narrows richEntry to
         non-null for the <component :is> inside — no non-null assertion. -->
    <template v-else-if="item.seg.type === 'rich'">
      <template v-if="item.richEntry">
        <component
          :is="item.richEntry.component"
          :payload="item.seg.payload"
          :synthesis="item.seg.synthesis"
        />
      </template>
      <!-- Unknown-tag fallback: render synthesis or content as markup. -->
      <div
        v-else
        class="speech-form__text"
        v-html="renderMarkup(item.seg.synthesis ?? item.seg.content ?? '')"
      />
    </template>
  </template>
</template>
