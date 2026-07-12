<!-- Folds consecutive superseded ACT cycles into one collapsible block. -->
<script setup lang="ts">
import { computed, ref } from 'vue';

const props = defineProps<{
  summaries: { tool_name: string; summary: string; state?: string; ended_at?: string | null }[];
}>();

const expanded = ref(false);

// Any errored step in the fold surfaces its error colour even while collapsed —
// a refetched error must not read as a benign step.
const hasError = computed(() => props.summaries.some((s) => s.state === 'error'));
</script>

<template>
  <div class="act-group" :class="{ 'act-group--expanded': expanded, 'act-tool--error': hasError }">
    <!-- Trace-pill toggle — always rendered so an expanded group can be collapsed
         again (matches BubbleFooter's trace pill); the summaries list below opens
         and closes with it. -->
    <button
      class="trace-pill"
      :class="{ 'trace-pill--open': expanded }"
      type="button"
      :aria-expanded="expanded"
      :aria-label="expanded ? 'Collapse steps' : 'Expand steps'"
      @click="expanded = !expanded"
    >
      <span class="trace-pill__dot" aria-hidden="true" />
      {{ summaries.length }} tool{{ summaries.length === 1 ? '' : 's' }} used
    </button>

    <div v-if="expanded" class="act-group__content">
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
    </div>
  </div>
</template>
