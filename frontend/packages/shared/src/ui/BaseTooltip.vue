<script setup lang="ts">
defineProps<{ text: string }>();
</script>

<template>
  <span class="base-tooltip">
    <slot />
    <span class="base-tooltip__bubble" role="tooltip">{{ text }}</span>
  </span>
</template>

<style scoped lang="scss">
.base-tooltip {
  position: relative;
  display: inline-flex;
  &__bubble {
    position: absolute;
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%) translateY(-4px);
    padding: 0.25rem 0.5rem;
    background: var(--bs-body-color);
    color: var(--bs-body-bg);
    border-radius: var(--bs-border-radius);
    font-size: 0.75rem;
    white-space: nowrap;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.12s ease;
  }
  &:hover &__bubble {
    opacity: 1;
  }
}
</style>
