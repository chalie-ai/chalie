<!-- ActionCardHost — renders ACT-cycle rich-card responses OUTSIDE the turn
     feed so they never pollute sortedBlocks. Mounted once in App.vue. -->
<script setup lang="ts">
import { useActionCard } from '../../composables/useActionCard';
import { renderMarkup } from '../../composables/useMarkup';

const { cards } = useActionCard();
</script>

<template>
  <div v-if="cards.length" class="action-card-host">
    <div
      v-for="card in cards"
      :key="card.id"
      class="action-card"
      :class="`action-card--${card.status}`"
    >
      <div v-if="card.status === 'acting'" class="action-card__acting">
        <span class="action-card__spinner" aria-hidden="true" />
        <span class="action-card__label">Acting…</span>
      </div>
      <div
        v-else-if="card.content"
        class="action-card__content chalie-markup"
        v-html="renderMarkup(card.content)"
      />
    </div>
  </div>
</template>

<style scoped lang="scss">
.action-card-host {
  position: fixed;
  bottom: calc(var(--dock-height, 120px) + 12px);
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  flex-direction: column;
  gap: 8px;
  z-index: 110;
  max-width: min(560px, 92vw);
  width: 100%;
  pointer-events: none;
}

.action-card {
  background: var(--bg-surface-2);
  border: 1px solid var(--border-strong);
  border-radius: 12px;
  padding: var(--space-md);
  pointer-events: auto;
}

.action-card--acting {
  border-color: color-mix(in oklab, var(--violet) 30%, transparent);
}

.action-card__acting {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--text-secondary);
  font-size: 13px;
}

.action-card__spinner {
  width: 14px;
  height: 14px;
  border: 2px solid color-mix(in oklab, var(--violet) 20%, transparent);
  border-top-color: var(--violet);
  border-radius: 50%;
  animation: ac-spin 0.7s linear infinite;
  flex-shrink: 0;
}

@keyframes ac-spin {
  to { transform: rotate(360deg); }
}

.action-card__content {
  font-size: 14px;
  color: var(--text-primary);
}
</style>
