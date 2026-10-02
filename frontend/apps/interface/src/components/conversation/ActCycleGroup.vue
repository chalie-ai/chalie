<!-- A footerless exchange's tool calls, behind the same pill and card the footer uses. -->
<script setup lang="ts">
import { ref } from 'vue';
import type { ConversationMessage } from '../../api/conversation';
import ToolCallList from './ToolCallList.vue';

defineProps<{ summaries: NonNullable<ConversationMessage['tool_calls']> }>();

const expanded = ref(false);
</script>

<template>
  <div class="act-group">
    <!-- Trace-pill toggle — always rendered so an expanded group can be collapsed
         again (matches BubbleFooter's trace pill); the summaries list below opens
         and closes with it. -->
    <button
      class="trace-pill"
      :class="{ 'trace-pill--open': expanded }"
      type="button"
      :aria-expanded="expanded"
      :aria-label="expanded ? 'Collapse steps' : 'Expand steps'"
      @click="expanded = !expanded"
    >
      <span class="trace-pill__dot" aria-hidden="true" />
      {{ summaries.length }} tool{{ summaries.length === 1 ? '' : 's' }} used
    </button>

    <!-- inert while folded, as on BubbleFooter: a delegate call row is a button,
         and the fold only collapses its height. -->
    <div class="trace-body" :class="{ 'trace-body--open': expanded }" :inert="!expanded">
      <div class="trace-body__inner">
        <ToolCallList :calls="summaries" />
      </div>
    </div>
  </div>
</template>
