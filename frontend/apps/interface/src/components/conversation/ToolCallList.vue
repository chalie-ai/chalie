<!-- One plain borderless row per tool call: a status icon, then the summary;
     a call that spawned a delegate turn ends with the robot icon and is a
     button into its transcript. -->
<script setup lang="ts">
import { Bot, Check, Minus, X } from '@lucide/vue';
import { delegatePillAttrs } from '../../composables/useDelegatePill';
import { callLabel, type StepCall } from '../../utils/turnSteps';

defineProps<{ calls: StepCall[] }>();

const STATUS_LABELS: Record<StepCall['state'], string> = { done: 'succeeded', error: 'failed', started: 'running' };
</script>

<template>
  <div class="calls">
    <component
      :is="c.delegate ? 'button' : 'div'"
      v-for="(c, i) in calls"
      :key="i"
      v-bind="delegatePillAttrs(c.delegate)"
      class="call"
      :class="`call--${c.state}`"
    >
      <component
        :is="c.state === 'done' ? Check : c.state === 'error' ? X : Minus"
        class="call__mark"
        :size="16"
        role="img"
        :aria-label="STATUS_LABELS[c.state]"
      />
      <span class="call__summary">{{ callLabel(c) }}</span>
      <Bot v-if="c.delegate" class="delegate-pill__icon" :size="16" aria-hidden="true" />
    </component>
  </div>
</template>
