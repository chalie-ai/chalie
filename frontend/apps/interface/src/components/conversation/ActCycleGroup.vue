<!-- Folds consecutive superseded ACT cycles into one collapsible block. -->
<script setup lang="ts">
import { ref, computed } from 'vue';
import { ChevronRight } from '@lucide/vue';

const props = defineProps<{ summaries: { tool_name: string; summary: string }[] }>();

const expanded = ref(false);

const preview = computed(() => {
  const first = props.summaries[0];
  return first?.summary || first?.tool_name || '';
});
</script>

<template>
  <div class="act-group" :class="{ 'act-group--expanded': expanded }">
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
      <span
        v-if="!expanded"
        class="act-group__preview"
        @click="expanded = true"
      >{{ preview }}</span>
      <template v-else>
        <div v-for="(s, i) in summaries" :key="i" class="act-cycle act-cycle--collapsed">
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
