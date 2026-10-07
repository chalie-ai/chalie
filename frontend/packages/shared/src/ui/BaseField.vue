<script setup lang="ts">
const model = defineModel<string>();
defineProps<{ label?: string; placeholder?: string; type?: string; id?: string }>();
</script>

<template>
  <label class="base-field">
    <span v-if="label" class="lbl">{{ label }}</span>
    <input
      :id="id"
      v-model="model"
      class="base-field__input"
      :type="type ?? 'text'"
      :placeholder="placeholder"
    />
  </label>
</template>

<style scoped>
.base-field { display: flex; flex-direction: column; gap: var(--space-xs); }
.base-field__input { padding: 0.5rem 0.75rem; }
</style>
