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

<style scoped>
.base-card { background: var(--surface); border: 1px solid var(--line); }
.base-card__head { padding: 0.75rem 1rem; border-bottom: 1px solid var(--line); font: 700 1.125rem/1.2 var(--font-display); }
.base-card__body { padding: 1rem; }
</style>
