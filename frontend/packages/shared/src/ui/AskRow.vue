<!-- One open ask as a row: where it came from, what it asks, how long it has
     waited. A permission is answered right here (Deny/Allow), and so is a
     sign-in code ask (its code typed in); a hand-off is answered with Done on
     its browser tab. A question has a card of its own (QuestionCard). Open is
     the caller's to act on: a hand-off always has one (its tab), a permission
     or a code ask only when it has a turn the user can open. -->
<script setup lang="ts">
import { computed } from 'vue';
import { useAsksStore, type RowAsk } from '../stores/asks';
import { actionLabel, CODE_ASK_WORDS, originLabel, waitedSeconds } from '../utils/askLabels';
import AskCodeEntry from './AskCodeEntry.vue';

const props = defineProps<{
  ask: RowAsk;
  /** The clock the "waiting for you" age is read against. */
  now: number;
}>();

const emit = defineEmits<{ open: [] }>();

const asks = useAsksStore();

const label = computed(() => {
  if (props.ask.kind === 'permission') return props.ask.data.summary || actionLabel(props.ask.data.action_id);
  if (props.ask.kind === 'code') return CODE_ASK_WORDS[props.ask.data.service].heading;
  return props.ask.data.reason || 'Chalie needs you in the browser';
});
</script>

<template>
  <div class="activity activity--docked">
    <span class="activity__lead">
      <span class="activity__mark" />
      <span v-if="ask.data.origin" class="activity__origin">{{ originLabel(ask.data.origin) }} · #{{ ask.data.origin.turn_id }} ·</span>
      <span class="activity__label">{{ label }}</span>
    </span>
    <span class="activity__controls">
      <span class="activity__wait">waiting for you · {{ waitedSeconds(ask.asked_at, now) }}s</span>
      <template v-if="ask.kind === 'permission'">
        <span v-if="ask.data.summary" class="activity__ask">{{ actionLabel(ask.data.action_id) }}</span>
        <span class="activity__answers">
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
        </span>
      </template>
      <AskCodeEntry v-else-if="ask.kind === 'code'" :ask="ask" />
      <span v-else-if="ask.kind === 'handoff'" class="activity__ask">Browser</span>
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
   action label ellipsizes. Controls wrap rather than run off a narrow screen. */
.activity__lead,
.activity__controls {
  display: inline-flex;
  align-items: center;
  gap: var(--space-sm);
  min-width: 0;
}

.activity__controls {
  flex-wrap: wrap;
  row-gap: var(--space-xs);
}

.activity__mark,
.activity__origin,
.activity__wait,
.activity__controls > button {
  flex-shrink: 0;
  white-space: nowrap;
}

/* Stop and Open are quiet text buttons that fill on hover. */
.activity__stop {
  display: inline-flex;
  align-items: center;
  height: var(--control-h);
  margin-inline-start: var(--space-sm);
  padding: 0 var(--space-sm);
  border-radius: var(--radius);
  background: none;
  font: 500 var(--fs-body)/1 var(--font-ui);
  color: var(--muted);
  transition:
    background-color var(--dur-1) var(--ease-out),
    color var(--dur-1) var(--ease-out);
}

.activity__stop:hover {
  background: var(--surface-2);
  color: var(--text);
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

.activity__ask::before {
  content: '· ';
}

/* Deny and Allow are one pair with the regular button's metrics: closer to
   each other than to the words before them, and wrapping together with no indent. */
.activity__answers {
  display: inline-flex;
  gap: var(--space-xs);
  flex-shrink: 0;
}

.activity__answer {
  height: var(--control-h);
  padding: 0 var(--space-md);
  border-radius: var(--radius);
  font: 600 var(--fs-body)/1 var(--font-ui);
  cursor: pointer;
  transition:
    background-color var(--dur-1) var(--ease-out),
    filter var(--dur-1) var(--ease-out),
    translate var(--dur-1) var(--ease-out);
}

.activity__answer:active:not(:disabled) {
  translate: 0 1px;
}

/* A sign-in code ask's Send stays disabled until the code is complete. */
.activity__answer:disabled {
  opacity: 0.45;
}

/* Allow is the screen's primary action, so it takes the one accent. */
.activity__answer--allow {
  background: var(--pink);
  color: var(--on-pink);
}

.activity__answer--allow:hover:not(:disabled) {
  filter: brightness(1.1);
}

.activity__answer--deny {
  background: var(--surface-2);
  color: var(--text);
}

.activity__answer--deny:hover {
  background: var(--cell);
}

/* A docked ask (the chat dock, the Brain's sticky dock) is a raised card whose
   words read like the conversation and wrap instead of widening the page. */
.activity--docked {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  column-gap: var(--space-sm);
  row-gap: var(--space-xs);
  min-width: 0;
  padding: var(--space-xs) var(--space-xs) var(--space-xs) var(--space-md);
  border-radius: var(--radius-card);
  background: var(--surface-2);
}

/* The card is Deny's own ground, so Deny steps up one fill to stay a button,
   and Open fills on hover to that same step. */
.activity--docked .activity__answer--deny,
.activity--docked .activity__stop:hover {
  background: var(--cell);
}

.activity--docked .activity__answer--deny:hover {
  background: color-mix(in srgb, var(--cell), var(--text) 10%);
}

.activity--docked .activity__label {
  font: 400 var(--fs-body)/1.55 var(--font-ui);
  color: var(--text);
  white-space: normal;
}

/* Wider than a phone, the answers sit at the card's right edge. */
@media (min-width: 641px) {
  .activity--docked .activity__controls {
    margin-inline-start: auto;
  }
}
</style>
