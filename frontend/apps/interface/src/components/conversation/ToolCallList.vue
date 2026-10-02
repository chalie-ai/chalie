<!-- One row per tool call: status dot, tool-name chip and summary, in one card.
     A call that spawned a delegate turn is a button into its transcript. -->
<script setup lang="ts">
import { Bot } from '@lucide/vue';
import type { ConversationMessage } from '../../api/conversation';
import { delegatePillAttrs } from '../../composables/useDelegatePill';

type ToolCall = NonNullable<ConversationMessage['tool_calls']>[number];

defineProps<{ calls: ToolCall[] }>();

const STATUS_LABELS: Record<ToolCall['state'], string> = { done: 'succeeded', error: 'failed', started: 'running' };
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
      <span class="call__dot" role="img" :aria-label="STATUS_LABELS[c.state]" />
      <span class="call__fn"><Bot v-if="c.delegate" class="delegate-pill__icon" :size="14" aria-hidden="true" />{{ c.tool_name }}</span>
      <span class="call__summary">{{ c.summary }}</span>
    </component>
  </div>
</template>
