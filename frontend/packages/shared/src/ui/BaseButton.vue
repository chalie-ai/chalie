<script setup lang="ts">
withDefaults(
  defineProps<{ variant?: 'primary' | 'secondary'; disabled?: boolean; type?: 'button' | 'submit' }>(),
  { variant: 'primary', disabled: false, type: 'button' },
);
defineEmits<{ click: [MouseEvent] }>();
</script>

<template>
  <button
    :type="type"
    :disabled="disabled"
    :class="['btn', { 'btn-primary': variant === 'primary' }]"
    @click="(e) => $emit('click', e)"
  >
    <slot />
  </button>
</template>
