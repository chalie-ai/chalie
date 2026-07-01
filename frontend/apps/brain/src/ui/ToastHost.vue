<script setup lang="ts">
import { useToast } from '../composables/useToast';

const { toasts, dismiss } = useToast();
</script>

<template>
  <div class="toast-host">
    <div v-for="toast in toasts" :key="toast.id" :class="['toast', `toast-${toast.type}`]">
      {{ toast.message }}
      <button
        v-if="toast.action && toast.onAction"
        class="toast-action"
        @click="
          toast.onAction?.();
          dismiss(toast.id);
        "
      >
        {{ toast.action }}
      </button>
    </div>
  </div>
</template>
