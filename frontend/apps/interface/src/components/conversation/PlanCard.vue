<!-- The plan the agent keeps while it works one message: a head counting the
     done tasks over a bar lit segment by segment, then its tasks in order, each
     with a mark that moves with the task — a dim tile to do, a glowing pink
     tile in progress, a check drawn in when done, a minus drawn in (the title
     struck through) when cancelled. Like the weather card it sits on a neon
     ground. Live it stands sticky under the message it plans; settled it opens
     from the footer's plan link. -->
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
      <span class="plan-card__label">Plan</span>
      <span class="plan-card__count" aria-hidden="true">{{ doneCount }}/{{ tasks.length }}</span>
      <span class="plan-card__summary">{{ doneCount }} of {{ tasks.length }} done<template v-if="cancelledCount">, {{ cancelledCount }} cancelled</template></span>
    </p>
    <div class="plan-card__bar" aria-hidden="true">
      <i v-for="t in tasks" :key="t.number" :class="`plan-card__segment--${t.status}`" />
    </div>
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
              width="14"
              height="14"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="3"
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
// Neon ground, opaque so the sticky card hides the thread scrolling under it.
// Its strength is the most that keeps the light theme's pink label at 4.5:1.
.plan-card {
  width: 100%;
  padding: 14px 16px 10px;
  background: linear-gradient(
    135deg,
    color-mix(in srgb, var(--pink) 14%, var(--bg)),
    color-mix(in srgb, var(--violet) 12%, var(--bg))
  );
}

.plan-card__head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  margin: 0;
}

.plan-card__label {
  font-family: var(--font-mono);
  font-size: var(--fs-mono);
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--pink-text);
}

.plan-card__count {
  font-family: var(--font-display);
  font-size: var(--fs-title);
  color: var(--text);
}

// One segment per task: lit and glowing once done, fading in while in
// progress, the dim track otherwise.
.plan-card__bar {
  display: flex;
  height: 2px;
  margin: 8px 0;
  background: color-mix(in srgb, var(--text) 12%, transparent);

  i {
    flex: 1;
    transition: box-shadow var(--dur-3) var(--ease-out);
  }
}

.plan-card__segment--done {
  background: var(--pink);
  box-shadow: 0 0 8px var(--pink);
}

.plan-card__segment--in_progress {
  background: linear-gradient(90deg, var(--pink), transparent);
}

.plan-card__tasks {
  margin: 0;
  padding: 0;
  list-style: none;
}

// The in-progress row's band runs to the card's edges.
.plan-task {
  display: flex;
  align-items: center;
  gap: 12px;
  min-height: 34px;
  margin: 0 -16px;
  padding: 0 16px;
  font-size: var(--fs-body);
  color: var(--muted);
  transition: color var(--dur-2) var(--ease-out), background-color var(--dur-2) var(--ease-out);
}

.plan-task--in_progress {
  background: color-mix(in srgb, var(--pink) 12%, transparent);
  font-weight: 500;
  color: var(--text);
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
  filter: drop-shadow(0 0 6px color-mix(in srgb, var(--allow-text) 70%, transparent));
}

.plan-task--cancelled .plan-task__mark {
  background: color-mix(in srgb, var(--text) 14%, transparent);
}

.plan-task__square {
  width: 16px;
  height: 16px;
  background: color-mix(in srgb, var(--text) 14%, transparent);
}

.plan-task--in_progress .plan-task__square {
  background: var(--pink);
  box-shadow: 0 0 14px 2px color-mix(in srgb, var(--pink) 80%, transparent);
  animation: plan-glow 1.4s ease-in-out infinite;
}

.plan-task__mark path {
  stroke-dasharray: 1;
  stroke-dashoffset: 0;
}

// Read by screen readers only — the count and the marks carry it visually.
.plan-card__summary,
.plan-task__status {
  position: absolute;
  width: 1px;
  height: 1px;
  overflow: hidden;
  clip-path: inset(50%);
  white-space: nowrap;
}

// A mark that just changed: a square pops in, a check or minus draws its
// stroke. Declared after the in-progress glow so the pop wins while it runs.
.plan-mark-enter-active.plan-task__square {
  animation: plan-pop var(--dur-2) var(--ease-out);
}

.plan-mark-enter-active path {
  animation: plan-draw var(--dur-3) var(--ease-out);
}

@keyframes plan-glow {
  50% { box-shadow: 0 0 4px color-mix(in srgb, var(--pink) 40%, transparent); }
}

@keyframes plan-pop {
  from { scale: 0; }
}

@keyframes plan-draw {
  from { stroke-dashoffset: 1; }
}
</style>
