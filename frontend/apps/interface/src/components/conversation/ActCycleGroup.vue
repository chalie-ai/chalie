<!-- Folds consecutive superseded ACT cycles into one collapsible block. -->
<script setup lang="ts">
import { computed, ref } from 'vue';
import { ChevronRight } from '@lucide/vue';

const props = defineProps<{
  summaries: { tool_name: string; summary: string; state?: string; ended_at?: string | null }[];
}>();

const expanded = ref(false);

const preview = computed(() => {
  const first = props.summaries[0];
  return first?.summary || first?.tool_name || '';
});

// Any errored step in the fold surfaces its error colour even while collapsed —
// a refetched error must not read as a benign step.
const hasError = computed(() => props.summaries.some((s) => s.state === 'error'));
</script>

<template>
  <div class="act-group" :class="{ 'act-group--expanded': expanded, 'act-tool--error': hasError }">
    <!-- Caret in a fixed left gutter so it aligns with the first content line. -->
    <button
      class="act-group__toggle"
      type="button"
      :aria-expanded="expanded"
      :aria-label="expanded ? 'Collapse steps' : 'Expand steps'"
      @click="expanded = !expanded"
    >
      <ChevronRight class="act-group__caret" :size="12" aria-hidden="true" />
    </button>

    <div class="act-group__content">
      <span v-if="!expanded" class="act-group__preview" @click="expanded = true">{{
        preview
      }}</span>
      <template v-else>
        <div
          v-for="(s, i) in summaries"
          :key="i"
          class="act-cycle act-cycle--collapsed"
          :class="{ 'act-tool--error': s.state === 'error' }"
        >
          <div class="act-summaries">
            <span class="act-tool__summary act-tool__summary--collapsed">
              {{ s.summary || s.tool_name }}
            </span>
          </div>
        </div>
      </template>
    </div>
  </div>
</template>
