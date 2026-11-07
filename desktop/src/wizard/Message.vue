<script setup lang="ts">
withDefaults(defineProps<{ text: string; detail?: string; tone?: 'error' | 'info' }>(), {
  detail: '',
  tone: 'info',
});
</script>

<template>
  <p class="message" :class="tone" role="status" aria-live="polite">
    {{ text }}
    <small v-if="detail">{{ detail }}</small>
  </p>
</template>
