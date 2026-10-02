<!-- One row per tool call: status dot, tool-name chip and summary, in one card. -->
<script setup lang="ts">
import type { ConversationMessage } from '../../api/conversation';

type ToolCall = NonNullable<ConversationMessage['tool_calls']>[number];

defineProps<{ calls: ToolCall[] }>();

const STATUS_LABELS: Record<ToolCall['state'], string> = { done: 'succeeded', error: 'failed', started: 'running' };
</script>

<template>
  <div class="calls">
    <div v-for="(c, i) in calls" :key="i" class="call" :class="`call--${c.state}`">
      <span class="call__dot" role="img" :aria-label="STATUS_LABELS[c.state]" />
      <span class="call__fn">{{ c.tool_name }}</span>
      <span class="call__summary">{{ c.summary }}</span>
    </div>
  </div>
</template>
