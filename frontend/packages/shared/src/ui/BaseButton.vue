<script setup lang="ts">
withDefaults(
  defineProps<{ variant?: 'primary' | 'secondary' | 'ghost'; disabled?: boolean; type?: 'button' | 'submit' }>(),
  { variant: 'primary', disabled: false, type: 'button' },
);
defineEmits<{ click: [MouseEvent] }>();
</script>

<template>
  <button
    :type="type"
    :disabled="disabled"
    :class="['base-btn', `base-btn--${variant}`]"
    @click="(e) => $emit('click', e)"
  >
    <slot />
  </button>
</template>

<style scoped lang="scss">
.base-btn {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  border: 1px solid transparent;
  border-radius: var(--bs-border-radius);
  cursor: pointer;
  transition: filter 0.15s ease, background 0.15s ease;
  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  &:hover:not(:disabled) {
    filter: brightness(1.1);
  }
  &--primary {
    background: var(--bs-primary);
    color: #fff;
  }
  &--secondary {
    background: var(--bs-secondary);
    color: #fff;
  }
  &--ghost {
    background: transparent;
    border-color: var(--bs-border-color);
    color: var(--bs-body-color);
  }
}
</style>
