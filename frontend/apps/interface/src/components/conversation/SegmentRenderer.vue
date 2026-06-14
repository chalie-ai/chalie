<script setup lang="ts">
import { computed } from 'vue';
import type { ConversationSegment } from '../../api/conversation';
import { renderMarkup } from '../../composables/useMarkup';
import { resolveRichCard } from '../rich/richRegistry';
import type { RichCardEntry } from '../rich/richRegistry';

const props = defineProps<{ segments: ConversationSegment[] }>();

// FIX 10: precompute resolved entries ONCE per segment so the template
// never calls resolveRichCard() twice for the same segment and the
// non-null assertion (!) is eliminated.
interface ResolvedSegment {
  seg: ConversationSegment;
  richEntry: RichCardEntry | null;
}

const resolved = computed<ResolvedSegment[]>(() =>
  props.segments.map((seg) => ({
    seg,
    richEntry: seg.type === 'rich' ? (resolveRichCard(seg.tag ?? '') ?? null) : null,
  }))
);
</script>

<template>
  <template v-for="(item, idx) in resolved" :key="idx">
    <!-- Text segment: render markup via v-html -->
    <div
      v-if="item.seg.type === 'text'"
      class="speech-form__text"
      v-html="renderMarkup(item.seg.content ?? '')"
    />

    <!-- Rich segment: use precomputed entry to avoid double resolve + unsafe ! -->
    <template v-else-if="item.seg.type === 'rich'">
      <component
        :is="item.richEntry!.component"
        v-if="item.richEntry"
        :payload="item.seg.payload"
        :synthesis="item.seg.synthesis"
      />
      <!-- Unknown-tag fallback: render synthesis or content as markup -->
      <div
        v-else
        class="speech-form__text"
        v-html="renderMarkup(item.seg.synthesis ?? item.seg.content ?? '')"
      />
    </template>
  </template>
</template>
