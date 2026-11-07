<!-- One step's work, set like the prose it sits above: its thinking as a
     "Thought for 9s" head that unfolds to the traces, then its tool calls,
     one row each and always in view. The running step's head reads
     "Thinking…" (or the ask it is parked on) and carries the turn's controls. -->
<script setup lang="ts">
import { computed, ref } from 'vue';
import { thought, type StepWork } from '../../utils/turnSteps';
import ToolCallList from './ToolCallList.vue';

const props = defineProps<{
  work: StepWork;
  /** Overrides the head — the ask a running step is parked on. */
  label?: string | null;
  live?: boolean;
}>();

const expanded = ref(false);
const traces = computed(() => props.work.thinking?.traces ?? []);
const head = computed(() => props.label || thought(props.work.thinking) || (props.live ? 'Thinking…' : ''));
</script>

<template>
  <div class="step">
    <div v-if="head" class="step__head">
      <button
        type="button"
        class="fold"
        :aria-expanded="traces.length ? expanded : undefined"
        :disabled="!traces.length"
        @click="expanded = !expanded"
      >
        <span v-if="live" class="activity__mark" />
        <span class="step__label">{{ head }}</span>
        <span v-if="traces.length" class="fold-chev" aria-hidden="true">›</span>
      </button>
      <slot />
    </div>
    <div
      v-if="traces.length"
      class="trace-body"
      :class="{ 'trace-body--open': expanded }"
      :inert="!expanded"
    >
      <div class="trace-body__inner">
        <pre v-for="(trace, i) in traces" :key="i" class="step__trace">{{ trace }}</pre>
      </div>
    </div>
    <ToolCallList v-if="work.tool_calls?.length" :calls="work.tool_calls" />
  </div>
</template>
