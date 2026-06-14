<script setup lang="ts">
import type { ActForm } from '../../stores/conversation';
import { useSessionStore } from '../../stores/session';

defineProps<{ form: ActForm }>();

const session = useSessionStore();

// FIX 7: wire stop button to the store's existing cancel action
function onStop(): void {
  void session.requestStop();
}
</script>

<template>
  <div class="act-cycle">
    <!-- Narration row: logo + narrative text + stop button -->
    <div class="act-row">
      <span class="act-logo" />
      <span class="act-narrative">{{ form.narration }}</span>
      <!-- FIX 7: stop button — aria-label, icon, and click handler match legacy renderer.js:226-232 -->
      <button
        class="act-stop-btn"
        aria-label="Stop and undo"
        title="Stop & undo"
        type="button"
        @click="onStop"
      >
        <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor">
          <path d="M4.5 2L1 5.5L4.5 9V6.5H10a3.5 3.5 0 0 1 0 7H7v2h3a5.5 5.5 0 0 0 0-11H4.5V2Z" />
        </svg>
      </button>
    </div>

    <!-- Tool pills — one per resolved/pending tool call -->
    <!-- FIX 11: use legacy class names (.act-tool + --running/--done/--error, NO act-tool-pill) -->
    <div class="act-tools">
      <div
        v-for="pill in form.tools"
        :key="pill.id"
        class="act-tool"
        :class="{
          'act-tool--running': !pill.resolved,
          'act-tool--done':    pill.resolved && pill.ok,
          'act-tool--error':   pill.resolved && !pill.ok,
        }"
        :data-call-id="pill.id"
      >
        <template v-if="pill.summary">
          <span class="act-tool__label">
            <span class="act-tool__name">{{ pill.name }}</span>
            <span class="act-tool__summary">— {{ pill.summary }}</span>
          </span>
        </template>
        <span v-else class="act-tool__name">{{ pill.name }}</span>

        <span class="act-tool__status">
          <span v-if="!pill.resolved" class="act-spinner" />
          <template v-else-if="pill.ok">
            {{ ((Math.max(0, pill.ms ?? 0)) / 1000).toFixed(1) }}s
          </template>
          <template v-else>error</template>
        </span>
      </div>
    </div>
  </div>
</template>
