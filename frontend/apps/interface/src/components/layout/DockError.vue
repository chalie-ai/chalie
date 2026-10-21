<!-- The closable error banner (session.errorMessage): rendered by every input
     dock, and by the thread panel when it shows a subagent transcript, which
     has no dock of its own. Its look is the global .dock-error rules. -->
<script setup lang="ts">
import { useSessionStore } from '../../stores/session';
import { AlertTriangle, X } from '@lucide/vue';

const session = useSessionStore();
</script>

<template>
  <div v-if="session.errorMessage" class="dock-error" role="alert">
    <AlertTriangle class="dock-error__icon" :size="16" aria-hidden="true" />
    <span class="dock-error__text">{{ session.errorMessage }}</span>
    <button
      class="dock-error__close"
      type="button"
      aria-label="Dismiss error"
      @click="session.errorMessage = null"
    >
      <X :size="16" />
    </button>
  </div>
</template>
