<!--
  ActCycleGroup — folds a run of consecutive superseded (collapsed) ACT cycles
  into one collapsible block. Collapsed by default: a chevron + the first
  summary line of the first cycle. Expanding rotates the chevron to a down caret
  and reveals the full act-trail, indented under the chevron.
-->
<script setup lang="ts">
import { ref, computed } from 'vue';
import type { ActForm } from '../../stores/conversation';
import ActCycle from './ActCycle.vue';

const props = defineProps<{ forms: ActForm[] }>();

const expanded = ref(false);

// Collapsed preview: the first summary line of the first cycle.
const preview = computed(() => {
  const pill = props.forms[0]?.tools[0];
  return pill ? (pill.summary || pill.name) : '';
});
</script>

<template>
  <div class="act-group" :class="{ 'act-group--expanded': expanded }">
    <button
      class="act-group__toggle"
      type="button"
      :aria-expanded="expanded"
      @click="expanded = !expanded"
    >
      <svg class="act-group__caret" width="12" height="12" viewBox="0 0 16 16" fill="none" aria-hidden="true">
        <path d="M6 4l4 4-4 4" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
      </svg>
      <span v-if="!expanded" class="act-group__preview">{{ preview }}</span>
    </button>

    <div v-if="expanded" class="act-group__body">
      <ActCycle v-for="f in forms" :key="f.id" :form="f" />
    </div>
  </div>
</template>
