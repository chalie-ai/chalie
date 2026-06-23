<script setup lang="ts">
defineProps<{ open: boolean; title?: string }>();
defineEmits<{ close: [] }>();
</script>

<template>
  <Teleport to="body">
    <Transition name="modal-fade">
      <div v-if="open" class="base-modal__backdrop" @click.self="$emit('close')">
        <div class="base-modal" role="dialog" aria-modal="true">
          <header v-if="title" class="base-modal__head">
            <h2>{{ title }}</h2>
            <button class="base-modal__x" aria-label="Close" @click="$emit('close')">×</button>
          </header>
          <div class="base-modal__body"><slot /></div>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<style scoped lang="scss">
.base-modal__backdrop {
  position: fixed;
  inset: 0;
  display: grid;
  place-items: center;
  background: rgba(0, 0, 0, 0.5);
  z-index: 1000;
}
.base-modal {
  background: var(--bs-card-bg);
  border: 1px solid var(--bs-card-border-color);
  border-radius: var(--bs-border-radius-lg);
  min-width: 320px;
  max-width: 90vw;
  &__head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.75rem 1rem;
    border-bottom: 1px solid var(--bs-card-border-color);
    h2 {
      margin: 0;
      font-size: 1rem;
    }
  }
  &__x {
    background: none;
    border: none;
    font-size: 1.25rem;
    cursor: pointer;
    color: var(--bs-body-color);
  }
  &__body {
    padding: 1rem;
  }
}
.modal-fade-enter-active,
.modal-fade-leave-active {
  transition: opacity 0.15s ease;
}
.modal-fade-enter-from,
.modal-fade-leave-to {
  opacity: 0;
}
</style>
