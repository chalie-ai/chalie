<script setup lang="ts">
const model = defineModel<string>();
defineProps<{ label?: string; placeholder?: string; type?: string; id?: string }>();
</script>

<template>
  <label class="base-field">
    <span v-if="label" class="base-field__label">{{ label }}</span>
    <input
      :id="id"
      v-model="model"
      class="base-field__input"
      :type="type ?? 'text'"
      :placeholder="placeholder"
    />
  </label>
</template>

<style scoped lang="scss">
.base-field {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  &__label {
    font-size: 0.8125rem;
    color: var(--bs-body-color);
    opacity: 0.8;
  }
  &__input {
    padding: 0.5rem 0.75rem;
    background: var(--bs-body-bg);
    color: var(--bs-body-color);
    border: 1px solid var(--bs-border-color);
    border-radius: var(--bs-border-radius);
    &:focus {
      outline: none;
      border-color: var(--bs-primary);
    }
  }
}
</style>
