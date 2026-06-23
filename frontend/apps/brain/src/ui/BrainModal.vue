<script setup lang="ts">
const props = defineProps<{
  modelValue: boolean;
  size?: 'sm' | 'lg' | '';
}>();

const emit = defineEmits<{
  (e: 'update:modelValue', value: boolean): void;
}>();

function close(): void {
  emit('update:modelValue', false);
}
</script>

<template>
  <Teleport to="body">
    <div v-if="props.modelValue" class="modal-overlay" @click.self="close">
      <div :class="['modal', props.size ? `modal-${props.size}` : '']">
        <slot :close="close" />
      </div>
    </div>
  </Teleport>
</template>
