<!-- A question's controls: the question, one button per answer with its
     description under it, and a field for the user's own answer, always on
     offer. On a single-select question a click on an answer sends it; on a
     multi-select one clicks pick and unpick, and Send sends the answers picked
     in the order they are listed. The typed answer goes with its own Send or
     Enter; it is kept in the store, so a row that comes back after an answer
     fails to send still has it. -->
<script setup lang="ts">
import { computed, ref, useId } from 'vue';
import { useAsksStore, type QuestionAsk } from '../stores/asks';

const props = defineProps<{ ask: QuestionAsk }>();

const asks = useAsksStore();
const questionId = useId();
const picked = ref<string[]>([]);
const other = computed({
  get: () => asks.otherDrafts[props.ask.ask_id] ?? '',
  set: (text: string) => asks.typeOther(props.ask.ask_id, text),
});
const typed = computed(() => other.value.trim());

function choose(label: string): void {
  if (!props.ask.data.multi_select) {
    void asks.reply(props.ask.ask_id, { labels: [label] });
    return;
  }
  picked.value = picked.value.includes(label)
    ? picked.value.filter((l) => l !== label)
    : [...picked.value, label];
}

function sendPicked(): void {
  const labels = props.ask.data.answers.map((a) => a.label).filter((l) => picked.value.includes(l));
  void asks.reply(props.ask.ask_id, { labels });
}

function sendOther(): void {
  void asks.reply(props.ask.ask_id, { other: typed.value });
}
</script>

<template>
  <div class="question">
    <p :id="questionId" class="question__text">{{ ask.data.question }}</p>
    <div class="question__answers" role="group" :aria-labelledby="questionId">
      <button
        v-for="answer in ask.data.answers"
        :key="answer.label"
        type="button"
        class="question__answer"
        :class="{ 'question__answer--picked': picked.includes(answer.label) }"
        :aria-pressed="ask.data.multi_select ? picked.includes(answer.label) : undefined"
        @click="choose(answer.label)"
      >
        <span class="question__label">{{ answer.label }}</span>
        <span v-if="answer.description" class="question__description">{{
          answer.description
        }}</span>
      </button>
      <button
        v-if="ask.data.multi_select"
        type="button"
        class="question__send"
        aria-label="Send the answers picked"
        :disabled="!picked.length"
        @click="sendPicked"
      >
        Send
      </button>
    </div>
    <form class="question__other" @submit.prevent="sendOther">
      <input
        v-model="other"
        type="text"
        class="question__field"
        placeholder="Other…"
        aria-label="Your own answer"
      />
      <button
        type="submit"
        class="question__send"
        aria-label="Send your own answer"
        :disabled="!typed"
      >
        Send
      </button>
    </form>
  </div>
</template>

<style scoped>
/* Below its row's lead and controls, on a line of its own in a wrapping head
   too; it takes the row's type, as the permission row's words and buttons do. */
.question {
  display: flex;
  flex-direction: column;
  gap: var(--space-sm);
  width: 100%;
  min-width: 0;
  margin-top: var(--space-sm);
}

.question__text {
  margin: 0;
  color: var(--text);
  overflow-wrap: anywhere;
}

.question__answers,
.question__other {
  display: flex;
  gap: var(--space-sm);
}

.question__answers {
  flex-wrap: wrap;
}

.question__answer {
  display: inline-flex;
  flex-direction: column;
  align-items: flex-start;
  justify-content: center;
  max-width: 100%;
  min-height: var(--control-h);
  padding: var(--space-xs) var(--space-sm);
  background: var(--surface-2);
  color: var(--text);
  font-weight: 500;
  text-align: left;
  overflow-wrap: anywhere;
  transition:
    background-color var(--dur-1) var(--ease-out),
    color var(--dur-1) var(--ease-out);
}

.question__answer:hover {
  background: var(--cell);
}

.question__answer--picked,
.question__answer--picked:hover {
  background: var(--pink);
  color: var(--on-pink);
}

.question__description {
  font-family: var(--font-mono);
  font-size: var(--fs-mono);
  font-weight: 400;
  color: var(--muted);
}

.question__answer--picked .question__description {
  color: inherit;
}

.question__field {
  flex: 1;
  min-width: 0;
}

.question__send {
  flex-shrink: 0;
  min-height: var(--control-h);
  padding: 0 var(--space-sm);
  background: var(--allow);
  color: var(--on-pink);
  font-weight: 500;
  transition:
    background-color var(--dur-1) var(--ease-out),
    color var(--dur-1) var(--ease-out);
}

.question__send:hover:not(:disabled) {
  background: var(--text);
  color: var(--bg);
}

.question__send:disabled {
  opacity: 0.45;
}
</style>
