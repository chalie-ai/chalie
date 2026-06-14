<script setup lang="ts">
import { useConfirm } from '../composables/useConfirm';

const { visible, pending, accept, cancel } = useConfirm();
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
          <button class="btn btn-secondary" @click="cancel">Cancel</button>
          <button :class="['btn', pending.confirmClass || 'btn-primary']" @click="accept">
            {{ pending.confirmLabel || 'Confirm' }}
          </button>
        </div>
      </div>
    </div>
  </Teleport>
</template>
