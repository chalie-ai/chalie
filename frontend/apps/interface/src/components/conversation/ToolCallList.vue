<!-- One plain borderless row per tool call: the tool's icon, the summary,
     then whether it succeeded. Each row opens to what the call did, under it. -->
<script setup lang="ts">
import { useId } from 'vue';
import { Check, Minus, X } from '@lucide/vue';
import type { LucideIcon } from '@lucide/vue';
import { callKind } from '../../utils/toolCatalog';
import { detailOpen, toggleDetail } from '../../utils/toolCallDetails';
import { callLabel, type StepCall } from '../../utils/turnSteps';
import ToolCallDetail from './ToolCallDetail.vue';

defineProps<{ calls: StepCall[] }>();

const STATUS: Record<StepCall['state'], { icon: LucideIcon; label: string }> = {
  done: { icon: Check, label: 'succeeded' },
  error: { icon: X, label: 'failed' },
  started: { icon: Minus, label: 'running' },
};

const listId = useId();
const detailId = (c: StepCall): string => `${listId}-call-${c.id}`;
</script>

<template>
  <div class="calls">
    <template v-for="c in calls" :key="c.id">
      <button
        type="button"
        class="call"
        :class="`call--${c.state}`"
        :aria-expanded="detailOpen(c.id)"
        :aria-controls="detailId(c)"
        @click="toggleDetail(c.id)"
      >
        <component :is="callKind(c).icon" class="call__icon" :size="16" aria-hidden="true" />
        <span class="call__summary">{{ callLabel(c) }}</span>
        <component
          :is="STATUS[c.state].icon"
          class="call__mark"
          :size="16"
          role="img"
          :aria-label="STATUS[c.state].label"
        />
      </button>
      <ToolCallDetail v-if="detailOpen(c.id)" :id="detailId(c)" :call="c" />
    </template>
  </div>
</template>
