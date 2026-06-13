<script setup lang="ts">
defineProps<{ title?: string }>();
</script>

<template>
  <section class="base-card">
    <header v-if="title || $slots.header" class="base-card__head">
      <slot name="header">{{ title }}</slot>
    </header>
    <div class="base-card__body"><slot /></div>
  </section>
</template>

<style scoped lang="scss">
.base-card {
  background: var(--bs-card-bg);
  border: 1px solid var(--bs-card-border-color);
  border-radius: var(--bs-border-radius-lg);
  overflow: hidden;
  &__head {
    padding: 0.75rem 1rem;
    background: var(--bs-card-cap-bg);
    border-bottom: 1px solid var(--bs-card-border-color);
    font-weight: 600;
  }
  &__body {
    padding: 1rem;
  }
}
</style>
