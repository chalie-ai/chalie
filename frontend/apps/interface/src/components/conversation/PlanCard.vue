<!-- The plan the agent keeps while it works one message: its tasks in order,
     each with a mark that moves with the task — a grey square to do, a pulsing
     pink square in progress, a check drawn in when done, a minus drawn in (the
     title struck through) when cancelled. Live it stands sticky under the
     message it plans; settled it opens from the footer's plan link. -->
<script setup lang="ts">
import { computed } from 'vue';
import type { PlanTask } from '../../api/conversation';

const props = defineProps<{ tasks: PlanTask[] }>();

const doneCount = computed(() => props.tasks.filter((t) => t.status === 'done').length);
const cancelledCount = computed(() => props.tasks.filter((t) => t.status === 'cancelled').length);

// The marks are drawn inline rather than as icon components: the draw-in
// animation needs `pathLength` on the path itself.
const CHECK_PATH = 'M20 6 9 17l-5-5';
const MINUS_PATH = 'M5 12h14';

// The mark is decoration; the status is read out with the title.
const STATUS_TEXT: Record<PlanTask['status'], string> = {
  todo: 'to do',
  in_progress: 'in progress',
  done: 'done',
  cancelled: 'cancelled',
};
</script>

<template>
  <section class="plan-card" aria-label="Plan">
    <p class="plan-card__head">
      <span class="plan-card__label">Plan</span> · {{ doneCount }} of {{ tasks.length }} done<template v-if="cancelledCount"> · {{ cancelledCount }} cancelled</template>
    </p>
    <ol class="plan-card__tasks">
      <li
        v-for="t in tasks"
        :key="t.number"
        class="plan-task"
        :class="`plan-task--${t.status}`"
      >
        <span class="plan-task__mark" aria-hidden="true">
          <!-- A status change swaps the mark, and the new one animates in; the
               first render does not animate, so a reload or an opened settled
               plan shows its marks still. -->
          <Transition name="plan-mark" mode="out-in" :duration="{ enter: 320, leave: 0 }">
            <svg
              v-if="t.status === 'done' || t.status === 'cancelled'"
              :key="t.status"
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
              stroke-linecap="round"
              stroke-linejoin="round"
            >
              <path pathLength="1" :d="t.status === 'done' ? CHECK_PATH : MINUS_PATH" />
            </svg>
            <span v-else :key="t.status" class="plan-task__square" />
          </Transition>
        </span>
        <span class="plan-task__title">{{ t.title }}</span>
        <span class="plan-task__status">, {{ STATUS_TEXT[t.status] }}</span>
      </li>
    </ol>
  </section>
</template>

<style scoped lang="scss">
.plan-card {
  width: min(100%, 520px);
  padding: 8px 15px 10px;
  background: var(--surface);
}

.plan-card__head {
  margin: 0;
  line-height: 24px;
  font-family: var(--font-mono);
  font-size: var(--fs-mono);
  letter-spacing: 0.02em;
  color: var(--muted);
}

.plan-card__label {
  text-transform: uppercase;
}

.plan-card__tasks {
  margin: 0;
  padding: 0;
  list-style: none;
}

.plan-task {
  display: flex;
  align-items: center;
  gap: 10px;
  min-height: 28px;
  font-size: var(--fs-body);
  color: var(--text);
  transition: color var(--dur-2) var(--ease-out);
}

.plan-task--done,
.plan-task--cancelled {
  color: var(--muted);
}

.plan-task--cancelled .plan-task__title {
  text-decoration: line-through;
}

.plan-task__mark {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 16px;
  height: 16px;
  flex-shrink: 0;
}

.plan-task--done .plan-task__mark {
  color: var(--allow-text);
}

.plan-task__square {
  width: 6px;
  height: 6px;
  background: var(--cell);
}

.plan-task--in_progress .plan-task__square {
  background: var(--pink);
  animation: working 1.4s ease-in-out infinite;
}

.plan-task__mark path {
  stroke-dasharray: 1;
  stroke-dashoffset: 0;
}

// Read by screen readers only — the mark carries the status visually.
.plan-task__status {
  position: absolute;
  width: 1px;
  height: 1px;
  overflow: hidden;
  clip-path: inset(50%);
  white-space: nowrap;
}

// A mark that just changed: a square pops in, a check or minus draws its
// stroke. Declared after the in-progress pulse so the pop wins while it runs.
.plan-mark-enter-active.plan-task__square {
  animation: plan-pop var(--dur-2) var(--ease-out);
}

.plan-mark-enter-active path {
  animation: plan-draw var(--dur-3) var(--ease-out);
}

@keyframes plan-pop {
  from { scale: 0; }
}

@keyframes plan-draw {
  from { stroke-dashoffset: 1; }
}
</style>
