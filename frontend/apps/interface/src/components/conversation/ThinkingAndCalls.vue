<!-- One step's work, one muted line under what it said: its thinking as a
     "Thought for 9s" head that unfolds to the traces, then what its tool
     calls did, "Wrote 2 files and read 1 file", that unfolds to one row per
     call. The running step's head reads "Thinking…" (or the ask it is parked
     on) and carries the turn's controls. -->
<script setup lang="ts">
import { computed, ref } from 'vue';
import { X } from '@lucide/vue';
import { callsOpen, setCallsOpen, summarizeCalls, thought, visibleCalls, type StepWork } from '../../utils/turnSteps';
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
const calls = computed(() => visibleCalls(props.work.tool_calls));
const summary = computed(() => summarizeCalls(calls.value));
const open = computed(() => callsOpen(calls.value));
</script>

<template>
  <div v-if="head || calls.length" class="step">
    <div class="step__head">
      <button
        v-if="head"
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
      <template v-if="calls.length">
        <span v-if="head" class="step__sep" aria-hidden="true">·</span>
        <button type="button" class="fold" :aria-expanded="open" @click="setCallsOpen(calls, !open)">
          <span class="step__summary">{{ summary.text }}</span>
          <template v-if="summary.failed">
            <span class="step__sep" aria-hidden="true">·</span>
            <span class="step__failed"><X :size="16" aria-hidden="true" />{{ summary.failed }} failed</span>
          </template>
          <span class="fold-chev" aria-hidden="true">›</span>
        </button>
      </template>
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
    <div v-if="calls.length" class="trace-body" :class="{ 'trace-body--open': open }" :inert="!open">
      <div class="trace-body__inner">
        <ToolCallList :calls="calls" />
      </div>
    </div>
  </div>
</template>
