<script setup lang="ts">
import { computed } from 'vue';
import { useConversationStore } from '../../stores/conversation';
import { useSessionStore } from '../../stores/session';

type CardKind = 'new' | 'thinking';

interface ActivityCard {
  turn_id: number | null;
  label: string;
  snippet: string;
  kind: CardKind;
}

const convo = useConversationStore();
const session = useSessionStore();

const cards = computed((): ActivityCard[] =>
  convo.threads
    .filter((t) => (t.unread ?? 0) > 0 || convo.isThreadActive(t.last_activity_at))
    .map((t) => ({
      turn_id: t.turn_id,
      label: t.gist ?? t.preview,
      snippet: t.preview,
      kind: ((t.unread ?? 0) > 0 ? 'new' : 'thinking') as CardKind,
    }))
    .sort((a, b) => {
      // newest-first; threads without a turn_id sort last
      if (a.turn_id === null && b.turn_id === null) return 0;
      if (a.turn_id === null) return 1;
      if (b.turn_id === null) return -1;
      return b.turn_id - a.turn_id;
    }),
);

function open(card: ActivityCard): void {
  if (card.turn_id === null) return;
  session.openThreadPanel(card.turn_id);
}
</script>

<template>
  <div v-if="cards.length" class="activity-sidebar">
    <div class="activity-header">
      <span class="header-dot" />
      <span class="header-label">New activity</span>
    </div>

    <button
      v-for="card in cards"
      :key="card.turn_id ?? card.label"
      :class="['activity-card', card.kind]"
      @click="open(card)"
    >
      <div class="card-top-row">
        <span class="card-label">{{ card.label }}</span>
        <span :class="['card-status-dot', card.kind]" />
      </div>
      <div class="card-snippet">{{ card.snippet }}</div>
    </button>
  </div>
</template>

<style scoped>
.activity-sidebar {
  position: fixed;
  top: 64px;
  right: 18px;
  width: 314px;
  /* Ambient rail: above the feed, below the footer dock / presence bar (100)
     and the thread panel (120) so a focused thread cleanly covers it. */
  z-index: 50;
  display: flex;
  flex-direction: column;
  gap: 10px;
  pointer-events: none;
  font-family: 'Inter', system-ui, sans-serif;
}

.activity-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 0 5px 1px;
}

.header-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--magenta);
  box-shadow: 0 0 8px color-mix(in oklab, var(--magenta) 60%, transparent);
  flex-shrink: 0;
}

.header-label {
  font: 600 10.5px Inter;
  letter-spacing: 0.07em;
  text-transform: uppercase;
  color: var(--text-tertiary);
}

.activity-card {
  border-radius: 13px;
  padding: 12px 13px;
  backdrop-filter: blur(10px);
  box-shadow: 0 16px 38px rgba(0, 0, 0, 0.55);
  cursor: pointer;
  pointer-events: auto;
  animation: cardEnter 0.3s var(--ease-out);
  text-align: left;
  border: none;
  width: 100%;
  display: block;
}

.activity-card.new {
  background: color-mix(in oklab, var(--magenta) 12%, var(--bg));
  border: 1px solid color-mix(in oklab, var(--magenta) 30%, transparent);
}

.activity-card.thinking {
  background: color-mix(in oklab, var(--violet) 10%, var(--bg));
  border: 1px solid color-mix(in oklab, var(--violet) 30%, transparent);
}

.card-top-row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 5px;
}

.card-label {
  font: 600 13px Inter;
  color: var(--text-primary);
  flex: 1;
  min-width: 0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.card-status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
}

.card-status-dot.new {
  background: var(--magenta);
  box-shadow: 0 0 10px color-mix(in oklab, var(--magenta) 60%, transparent);
}

.card-status-dot.thinking {
  background: var(--violet);
  box-shadow: 0 0 10px color-mix(in oklab, var(--violet) 70%, transparent);
  animation: pulseV 1.4s ease-in-out infinite;
}

.card-snippet {
  font: 400 11.5px / 1.45 Inter;
  color: var(--text-secondary);
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
  text-align: left;
}
</style>
