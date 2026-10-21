<!-- The plan the agent keeps while it works one message: a rounded card on a
     soft neon ground, a head counting the done tasks over a gapped bar, then
     its tasks in order, each with a mark that moves with the task — a dashed
     circle to do, a spinning arc in progress, a check-circle drawn in when
     done, a slashed circle (the title struck through) when cancelled. Live it
     stands sticky under the message it plans; settled it opens from the
     footer's plan link. -->
<script setup lang="ts">
import { computed } from 'vue';
import { CircleDashed } from '@lucide/vue';
import type { PlanTask } from '../../api/conversation';

const props = defineProps<{ tasks: PlanTask[] }>();

const doneCount = computed(() => props.tasks.filter((t) => t.status === 'done').length);
const cancelledCount = computed(() => props.tasks.filter((t) => t.status === 'cancelled').length);

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
               plan shows its marks still. The other marks are drawn inline:
               the draw-in animation needs `pathLength` on the glyph itself. -->
          <Transition name="plan-mark" mode="out-in" :duration="{ enter: 320, leave: 0 }">
            <CircleDashed v-if="t.status === 'todo'" key="todo" :size="16" />
            <svg
              v-else
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
              <template v-if="t.status === 'in_progress'">
                <circle class="plan-task__track" cx="12" cy="12" r="10" />
                <path class="plan-task__spin" d="M22 12a10 10 0 0 0-10-10" />
              </template>
              <template v-else>
                <circle cx="12" cy="12" r="10" />
                <path
                  pathLength="1"
                  class="plan-task__glyph"
                  :d="t.status === 'done' ? 'm8 12 3 3 5-6' : 'M8 16 16 8'"
                />
              </template>
            </svg>
          </Transition>
        </span>
        <span class="plan-task__title">{{ t.title }}</span>
        <span class="plan-task__status">, {{ STATUS_TEXT[t.status] }}</span>
      </li>
    </ol>
  </section>
</template>

<style scoped lang="scss">
// Soft neon ground, opaque so the sticky card hides the thread scrolling
// under it.
.plan-card {
  width: 100%;
  padding: 14px 16px 8px;
  border-radius: 12px;
  background: linear-gradient(
    135deg,
    color-mix(in srgb, var(--pink) 6%, var(--bg)),
    color-mix(in srgb, var(--violet) 6%, var(--bg))
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

// One segment per task: the dim track, solid pink once done, a pulsing
// pink while in progress, an outlined gap when cancelled. The states sit
// inside the bar so they outrank the track's colour.
.plan-card__bar {
  display: flex;
  gap: 4px;
  height: 3px;
  margin: 10px 0 6px;

  i {
    flex: 1;
    border-radius: 2px;
    background: var(--cell);
  }

  .plan-card__segment--done {
    background: var(--pink);
  }

  .plan-card__segment--in_progress {
    background: color-mix(in srgb, var(--pink) 45%, var(--cell));
    animation: plan-fade 1.6s ease-in-out infinite;
  }

  .plan-card__segment--cancelled {
    background: transparent;
    box-shadow: inset 0 0 0 1px var(--cell);
  }
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
  min-height: 32px;
  font-size: var(--fs-body);
  color: var(--text);
  transition: color var(--dur-2) var(--ease-out);
}

.plan-task--in_progress {
  font-weight: 500;
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

.plan-task--todo .plan-task__mark,
.plan-task--cancelled .plan-task__mark {
  color: var(--muted);
}

.plan-task--in_progress .plan-task__mark {
  color: var(--pink-text);
}

.plan-task--done .plan-task__mark {
  color: var(--allow-text);
}

.plan-task__track {
  opacity: 0.22;
}

.plan-task__spin {
  transform-origin: 12px 12px;
  animation: plan-spin 1s linear infinite;
}

.plan-task__glyph {
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

// A mark that just changed: the to-do and in-progress marks pop in, a done
// or cancelled glyph draws its stroke.
.plan-task--todo .plan-mark-enter-active,
.plan-task--in_progress .plan-mark-enter-active {
  animation: plan-pop var(--dur-2) var(--ease-out);
}

.plan-mark-enter-active .plan-task__glyph {
  animation: plan-draw var(--dur-3) var(--ease-out);
}

@keyframes plan-fade {
  50% { opacity: 0.45; }
}

@keyframes plan-spin {
  to { transform: rotate(360deg); }
}

@keyframes plan-pop {
  from { scale: 0; }
}

@keyframes plan-draw {
  from { stroke-dashoffset: 1; }
}
</style>
