<!-- The exchange's work, in row order, folded open BELOW the answer's footer:
     mid-turn prose (the answer row is excluded — it renders above) and tool
     calls. No 'thought' step here — a reply with no work at all shows its
     reasoning traces in the footer's own fallback list instead. -->
<script setup lang="ts">
import type { ConversationMessage } from '../../api/conversation';
import ChalieBubble from './ChalieBubble.vue';
import ToolCallList from './ToolCallList.vue';

defineProps<{
  /** The exchange's rows, in order — the work this list shows. */
  messages: ConversationMessage[];
  /** The answer row's id — excluded (it renders above the fold, not in it). */
  answerId: string | null;
}>();
</script>

<template>
  <div class="activity__steps">
    <template v-for="m in messages" :key="m.id">
      <ChalieBubble
        v-if="m.role === 'assistant' && m.content.trim() !== '' && String(m.id) !== answerId"
        :message="m"
        class="activity__prose"
      />
      <ToolCallList v-if="m.tool_calls?.length" :calls="m.tool_calls" />
    </template>
  </div>
</template>
