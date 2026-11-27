<script setup lang="ts">
import { onBeforeUnmount, onMounted } from 'vue';
import { useConfirm } from '../composables/useConfirm';

const { visible, pending, accept, cancel } = useConfirm();

// Escape closes only the topmost layer: a layer that handled it marks it with
// preventDefault. Listeners run in mount order (palette, this dialog, then App's
// mobile drawer), which is top to bottom on screen.
function onKeydown(e: KeyboardEvent): void {
  if (e.key !== 'Escape' || e.defaultPrevented || !visible.value) return;
  e.preventDefault();
  cancel();
}

onMounted(() => document.addEventListener('keydown', onKeydown));
onBeforeUnmount(() => document.removeEventListener('keydown', onKeydown));
</script>

<template>
  <Teleport to="body">
    <div v-if="visible && pending" class="modal-overlay" @click.self="cancel">
      <div class="modal modal-sm">
        <div class="modal-header">
          <h3>{{ pending.title }}</h3>
        </div>
        <p class="modal-desc">{{ pending.desc }}</p>
        <div class="modal-actions">
          <button class="btn" @click="cancel">Cancel</button>
          <button :class="['btn', pending.confirmClass || 'btn-primary']" @click="accept">
            {{ pending.confirmLabel || 'Confirm' }}
          </button>
        </div>
      </div>
    </div>
  </Teleport>
</template>
