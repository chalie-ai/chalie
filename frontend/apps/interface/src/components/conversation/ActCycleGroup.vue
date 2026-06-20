<!-- Folds consecutive superseded ACT cycles into one collapsible block. -->
<script setup lang="ts">
import { ref, computed } from 'vue';
import { ChevronRight } from '@lucide/vue';
import type { ActForm } from '../../stores/conversation';
import ActCycle from './ActCycle.vue';

const props = defineProps<{ forms: ActForm[] }>();

const expanded = ref(false);

const preview = computed(() => {
  const pill = props.forms[0]?.tools[0];
  return pill?.summary || pill?.name || '';
});
</script>

<template>
  <div class="act-group" :class="{ 'act-group--expanded': expanded }">
    <!-- Caret in a fixed left gutter so it aligns with the first content line
         in both states — no empty caret-only row above the trail. -->
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
        <ActCycle v-for="f in forms" :key="f.id" :form="f" />
      </template>
    </div>
  </div>
</template>
