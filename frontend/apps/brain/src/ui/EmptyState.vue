<script setup lang="ts">
const props = defineProps<{
  message?: string;
  icon?: string;
}>();
</script>

<template>
  <div class="empty-state">
    <div v-if="props.icon" class="empty-icon" v-html="props.icon"></div>
    <p>{{ props.message ?? 'Nothing here yet.' }}</p>
    <slot />
  </div>
</template>
