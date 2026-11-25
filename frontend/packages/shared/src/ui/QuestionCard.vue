<!-- Every open question on one card, a page each: the question as its title,
     one numbered row per answer with its description under it, and a last row
     for the user's own words. A click, a number key or Enter on a single-select
     question's answer sends it; a multi-select question's answers pick and
     unpick, and one button sends them all. Skip, Esc and the close button
     close the question unanswered. Answering or skipping a page shows the next.
     What is picked or typed lives in the store, so a question whose answer
     fails to send comes back as the user left it. A question no live line on
     screen shows also says where it came from, with Open. -->
<script setup lang="ts">
import { computed, nextTick, ref, useId, watch } from 'vue';
import { ChevronLeft, ChevronRight, Pencil, X } from '@lucide/vue';
import { useAsksStore, type QuestionAsk } from '../stores/asks';
import { originLabel, waitedSeconds } from '../utils/askLabels';

const props = defineProps<{
  questions: QuestionAsk[];
  /** The clock the "waiting for you" age is read against. */
  now: number;
}>();

const emit = defineEmits<{ open: [ask: QuestionAsk] }>();

const asks = useAsksStore();
const titleId = useId();
const countId = useId();
const rootRef = ref<HTMLElement | null>(null);
const index = ref(0);
let shownId: string | null = null;

const current = computed(() => props.questions[index.value] ?? null);
const draft = computed(() => (current.value ? asks.drafts[current.value.ask_id] : undefined));
const picked = computed(() => (draft.value && 'labels' in draft.value ? draft.value.labels : []));
const typed = computed(() => (draft.value && 'other' in draft.value ? draft.value.other : ''));
const ready = computed(() => picked.value.length > 0 || typed.value.trim() !== '');
const offLine = computed(() => current.value != null && (asks.shown[current.value.ask_id] ?? 0) <= 0);

// The page stays on the question it shows while others come and go; when that
// one leaves, the one now in its place shows.
watch(
  () => props.questions.map((q) => q.ask_id),
  (ids) => {
    const kept = shownId ? ids.indexOf(shownId) : -1;
    index.value = kept >= 0 ? kept : Math.min(index.value, Math.max(ids.length - 1, 0));
  },
);
watch(current, (q) => (shownId = q?.ask_id ?? null), { immediate: true });

/** Close this page — answered or skipped — and keep the keyboard on the card's next page. */
function leave(settle: (askId: string) => Promise<void>): void {
  const q = current.value;
  if (!q) return;
  const hadFocus = rootRef.value?.contains(document.activeElement) ?? false;
  void settle(q.ask_id);
  if (hadFocus) void nextTick(() => rootRef.value?.querySelector<HTMLElement>('.qcard__answer')?.focus());
}

function choose(label: string): void {
  const q = current.value;
  if (!q) return;
  if (!q.data.multi_select) {
    leave((id) => asks.reply(id, { labels: [label] }));
    return;
  }
  const next = picked.value.includes(label) ? picked.value.filter((l) => l !== label) : [...picked.value, label];
  asks.draft(q.ask_id, { labels: q.data.answers.map((a) => a.label).filter((l) => next.includes(l)) });
}

function type(text: string): void {
  if (current.value) asks.draft(current.value.ask_id, { other: text });
}

function send(): void {
  const words = typed.value.trim();
  if (words) leave((id) => asks.reply(id, { other: words }));
  else if (picked.value.length) leave((id) => asks.reply(id, { labels: picked.value }));
}

function skip(): void {
  leave((id) => asks.dismiss(id));
}

/** ↑/↓ walk the answers and the field. */
function move(step: number): void {
  const stops = [...(rootRef.value?.querySelectorAll<HTMLElement>('.qcard__answer, .qcard__field') ?? [])];
  const at = stops.indexOf(document.activeElement as HTMLElement);
  const to = at < 0 ? (step > 0 ? 0 : stops.length - 1) : Math.min(Math.max(at + step, 0), stops.length - 1);
  stops[to]?.focus();
}

function onKeydown(e: KeyboardEvent): void {
  const q = current.value;
  if (!q || e.isComposing || e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key === 'Escape') {
    // The card's own: a panel or page closing on Escape leaves it be.
    e.preventDefault();
    e.stopPropagation();
    skip();
  } else if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
    e.preventDefault();
    move(e.key === 'ArrowDown' ? 1 : -1);
  } else if (e.target instanceof HTMLInputElement) {
    if (e.key === 'Enter') {
      e.preventDefault();
      send();
    }
  } else if (/^[1-4]$/.test(e.key) && Number(e.key) <= q.data.answers.length) {
    e.preventDefault();
    choose(q.data.answers[Number(e.key) - 1].label);
  }
}
</script>

<template>
  <div v-if="current" ref="rootRef" class="qcard" @keydown="onKeydown">
    <div :key="current.ask_id" class="qcard__panel">
      <div class="qcard__head">
        <div class="qcard__heading">
          <p :id="titleId" class="qcard__title">{{ current.data.question }}</p>
          <p v-if="offLine" class="qcard__meta">
            <span v-if="current.data.origin">{{ originLabel(current.data.origin) }} · #{{ current.data.origin.turn_id }} ·</span>
            <span>waiting for you · {{ waitedSeconds(current.asked_at, now) }}s</span>
            <button v-if="current.data.origin" type="button" class="activity__stop" @click="emit('open', current)">Open</button>
          </p>
        </div>
        <div v-if="questions.length > 1" class="qcard__pager" role="group" :aria-label="`Question ${index + 1} of ${questions.length}`">
          <button type="button" class="qcard__icon" aria-label="Previous question" :disabled="index === 0" @click="index--">
            <ChevronLeft :size="16" aria-hidden="true" />
          </button>
          <span aria-hidden="true">{{ index + 1 }} of {{ questions.length }}</span>
          <button type="button" class="qcard__icon" aria-label="Next question" :disabled="index === questions.length - 1" @click="index++">
            <ChevronRight :size="16" aria-hidden="true" />
          </button>
        </div>
        <button type="button" class="qcard__icon" aria-label="Close" @click="skip">
          <X :size="16" aria-hidden="true" />
        </button>
      </div>

      <div class="qcard__answers" role="group" :aria-labelledby="titleId" :aria-describedby="countId">
        <span :id="countId" hidden>{{
          current.data.multi_select
            ? `Choose one or more of ${current.data.answers.length} options.`
            : `Choose one of ${current.data.answers.length} options.`
        }}</span>
        <button
          v-for="(answer, i) in current.data.answers"
          :key="answer.label"
          type="button"
          class="qcard__answer"
          :class="{ 'qcard__answer--picked': picked.includes(answer.label) }"
          :aria-pressed="current.data.multi_select ? picked.includes(answer.label) : undefined"
          @click="choose(answer.label)"
        >
          <span class="qcard__badge" aria-hidden="true">{{ i + 1 }}</span>
          <span class="qcard__words">
            <span class="qcard__label">{{ answer.label }}</span>
            <span v-if="answer.description" class="qcard__description">{{ answer.description }}</span>
          </span>
        </button>
      </div>

      <div class="qcard__other">
        <span class="qcard__badge" aria-hidden="true"><Pencil :size="16" /></span>
        <input
          class="qcard__field"
          type="text"
          placeholder="Something else"
          aria-label="Something else"
          :value="typed"
          @input="type(($event.target as HTMLInputElement).value)"
        />
        <button v-if="ready" type="button" class="btn btn-primary" @click="send">
          {{ questions.length > 1 ? 'Next' : 'Submit' }}
        </button>
        <button
          v-else
          type="button"
          class="btn"
          :title="current.data.multi_select ? 'Select an option to send' : 'Type an answer to send'"
          @click="skip"
        >
          Skip
        </button>
      </div>
    </div>
    <p class="qcard__hint">↑↓ to navigate · Enter to select · Esc to skip</p>
  </div>
</template>

<style scoped>
/* A dock's own type is the asks' muted mono; the card is read as body text. */
.qcard {
  width: 100%;
  font: 400 var(--fs-body) / 1.55 var(--font-ui);
  color: var(--text);
  animation: rise var(--dur-2) var(--ease-out);
}

.qcard__panel {
  display: flex;
  flex-direction: column;
  gap: var(--space-sm);
  padding: var(--space-md);
  background: var(--surface-2);
}

.qcard__head {
  display: flex;
  align-items: flex-start;
  gap: var(--space-xs);
}

.qcard__heading {
  flex: 1;
  min-width: 0;
  padding-right: var(--space-sm);
}

.qcard__title {
  margin: 0;
  font-size: var(--fs-title);
  font-weight: 600;
  line-height: var(--control-h);
  overflow-wrap: anywhere;
}

.qcard__meta {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  column-gap: var(--space-sm);
  margin: 0;
  font-family: var(--font-mono);
  font-size: var(--fs-mono);
  color: var(--muted);
}

.qcard__meta .activity__stop {
  height: auto;
}

.qcard__pager {
  display: inline-flex;
  flex-shrink: 0;
  align-items: center;
  font-family: var(--font-mono);
  font-size: var(--fs-mono);
  color: var(--muted);
  white-space: nowrap;
}

.qcard__icon {
  display: inline-flex;
  flex-shrink: 0;
  align-items: center;
  justify-content: center;
  width: var(--control-h);
  height: var(--control-h);
  padding: 0;
  background: none;
  color: var(--muted);
  transition:
    background-color var(--dur-1) var(--ease-out),
    color var(--dur-1) var(--ease-out);
}

.qcard__icon:hover:not(:disabled) {
  background: var(--cell);
  color: var(--text);
}

.qcard__icon:disabled {
  opacity: 0.45;
}

.qcard__answers {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

/* Rows part by space and fill, never a rule. */
.qcard__answer {
  display: flex;
  align-items: center;
  gap: 12px;
  width: 100%;
  min-height: 48px;
  padding: var(--space-sm);
  background: none;
  text-align: left;
  transition: background-color var(--dur-1) var(--ease-out);
}

.qcard__answer:hover,
.qcard__answer--picked {
  background: var(--cell);
}

.qcard__answer:focus-visible {
  outline-offset: -2px;
}

.qcard__badge {
  display: inline-flex;
  flex-shrink: 0;
  align-items: center;
  justify-content: center;
  width: var(--control-h);
  height: var(--control-h);
  background: var(--surface);
  font: 500 var(--fs-mono) / 1 var(--font-mono);
  color: var(--muted);
  transition:
    background-color var(--dur-1) var(--ease-out),
    color var(--dur-1) var(--ease-out);
}

.qcard__answer--picked .qcard__badge {
  background: var(--pink);
  color: var(--on-pink);
}

.qcard__words {
  display: flex;
  flex-direction: column;
  min-width: 0;
}

.qcard__label {
  font-weight: 500;
  overflow-wrap: anywhere;
}

.qcard__description {
  font-family: var(--font-mono);
  font-size: var(--fs-mono);
  color: var(--muted);
  overflow-wrap: anywhere;
}

.qcard__other {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: var(--space-sm);
  background: var(--surface);
}

.qcard__other .qcard__badge {
  background: var(--bg);
  color: var(--text);
}

.qcard__field {
  flex: 1;
  min-width: 0;
}

.qcard__hint {
  margin: 6px 0 0;
  font-family: var(--font-mono);
  font-size: var(--fs-mono);
  color: var(--muted);
  text-align: center;
}

/* Keys mean nothing on a touch screen. */
@media (hover: none) {
  .qcard__hint {
    display: none;
  }
}
</style>
