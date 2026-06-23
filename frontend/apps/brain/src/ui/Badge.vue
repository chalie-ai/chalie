<script setup lang="ts">
const props = defineProps<{
  variant?: string;
}>();
</script>

<template>
  <span :class="['badge', props.variant ? `badge-${props.variant}` : '']">
    <slot />
  </span>
</template>
