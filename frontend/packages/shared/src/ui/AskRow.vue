<!-- One open ask as a row: where it came from, what it asks, how long it has
     waited. A permission is answered right here (Deny/Allow); a hand-off is
     answered with Done on its browser tab. Open is the caller's to act on: a
     hand-off always has one (its tab), a permission only when it has a turn
     the user can open. -->
<script setup lang="ts">
import { computed } from 'vue';
import { useAsksStore, type Ask } from '../stores/asks';
import { actionLabel, originLabel, waitedSeconds } from '../utils/askLabels';

const props = defineProps<{
  ask: Ask;
  /** The clock the "waiting for you" age is read against. */
  now: number;
}>();

const emit = defineEmits<{ open: [] }>();

const asks = useAsksStore();

const label = computed(() =>
  props.ask.kind === 'permission'
    ? props.ask.data.summary || actionLabel(props.ask.data.action_id)
    : props.ask.data.reason || 'Chalie needs you in the browser',
);
</script>

<template>
  <div class="activity">
    <span class="activity__lead">
      <span class="activity__mark" />
      <span v-if="ask.data.origin" class="activity__origin">{{ originLabel(ask.data.origin) }} · #{{ ask.data.origin.turn_id }} ·</span>
      <span class="activity__label">{{ label }}</span>
    </span>
    <span class="activity__controls">
      <span class="activity__wait">· waiting for you · {{ waitedSeconds(ask.asked_at, now) }}s</span>
      <template v-if="ask.kind === 'permission'">
        <span v-if="ask.data.summary" class="activity__ask">{{ actionLabel(ask.data.action_id) }}</span>
        <button
          type="button"
          class="activity__answer activity__answer--deny"
          @click="asks.reply(ask.ask_id, false)"
        >
          Deny
        </button>
        <button
          type="button"
          class="activity__answer activity__answer--allow"
          @click="asks.reply(ask.ask_id, true)"
        >
          Allow
        </button>
      </template>
      <span v-else class="activity__ask">Browser</span>
      <button v-if="ask.kind === 'handoff' || ask.data.origin" type="button" class="activity__stop" @click="emit('open')">Open</button>
    </span>
  </div>
</template>

<style>
/* The chat app's live activity line (ActivityLine, ThinkingAndCalls) wears
   these classes too, so they are not scoped. */
.activity {
  animation: rise var(--dur-2) var(--ease-out);
}

/* The mark pulses pink while the turn works. */
.activity__mark {
  flex-shrink: 0;
  width: 6px;
  height: 6px;
  background: var(--pink);
  animation: working 1.4s ease-in-out infinite;
}

.activity__label {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: var(--text);
}

/* An ask's lead and a running step's controls each hold together; the
   action label ellipsizes. */
.activity__lead,
.activity__controls {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
}

.activity__mark,
.activity__origin,
.activity__wait,
.activity__controls > button {
  flex-shrink: 0;
  white-space: nowrap;
}

.activity__stop {
  display: inline-flex;
  align-items: center;
  height: var(--control-h);
  padding: 0;
  background: none;
  font-family: var(--font-mono);
  font-size: var(--fs-mono);
  color: var(--muted);
  text-decoration: underline;
  text-underline-offset: 3px;
  transition: color var(--dur-1) var(--ease-out);
}

.activity__stop:hover {
  color: var(--deny-text);
}

.activity__wait,
.activity__ask {
  overflow: hidden;
  text-overflow: ellipsis;
  font-family: var(--font-mono);
  font-size: var(--fs-mono);
  color: var(--muted);
  white-space: nowrap;
}

.activity__answer {
  min-height: var(--control-h);
  padding: 0 var(--space-sm);
  font-weight: 500;
  cursor: pointer;
  transition:
    background-color var(--dur-1) var(--ease-out),
    color var(--dur-1) var(--ease-out),
    translate var(--dur-1) var(--ease-out);
}

.activity__answer:active {
  translate: 0 1px;
}

.activity__answer--allow {
  background: var(--allow);
  color: var(--on-pink);
}

.activity__answer--allow:hover {
  background: var(--text);
  color: var(--bg);
}

.activity__answer--deny {
  background: var(--surface-2);
  color: var(--text);
}

.activity__answer--deny:hover {
  background: var(--cell);
}
</style>
