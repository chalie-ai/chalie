<script setup lang="ts">
import type { ActForm } from '../../stores/conversation';

defineProps<{ form: ActForm }>();
</script>

<template>
  <div class="act-cycle">
    <!-- Narration row: logo + narrative text -->
    <div class="act-row">
      <span class="act-logo" />
      <span class="act-narrative">{{ form.narration }}</span>
    </div>

    <!-- Tool pills — one per resolved/pending tool call -->
    <div class="act-tools">
      <div
        v-for="pill in form.tools"
        :key="pill.id"
        class="act-tool act-tool-pill"
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

<style scoped>
/*
 * actEntry keyframe — defined here because no actEntry exists in interface.scss.
 * Mirrors formEntry's shape (fade + small rise) from the shell keyframes.
 */
@keyframes actEntry {
  from { opacity: 0; transform: translateY(6px); }
  to   { opacity: 1; transform: translateY(0); }
}

.act-cycle {
  animation: actEntry 200ms var(--ease-out, cubic-bezier(0, 0, 0.2, 1)) both;
}
</style>
