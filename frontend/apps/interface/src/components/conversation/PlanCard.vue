<!-- The plan the agent keeps while it works one message: its tasks in order on
     a rounded card with a soft neon ground, each with a mark that moves with
     the task — a dashed circle to do, a spinning arc in progress, a
     check-circle drawn in when done, a slashed circle (the title struck
     through) when cancelled. Live it stands sticky under the message it plans;
     settled it opens from the footer's plan link. -->
<script setup lang="ts">
import { CircleDashed } from '@lucide/vue';
import type { PlanTask } from '../../api/conversation';

defineProps<{ tasks: PlanTask[] }>();

// The mark is decoration; the status is read out with the title.
const STATUS_TEXT: Record<PlanTask['status'], string> = {
  todo: 'to do',
  in_progress: 'in progress',
  done: 'done',
  cancelled: 'cancelled',
};
</script>

<template>
  <ol class="plan-card" aria-label="Plan">
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
          <CircleDashed v-if="t.status === 'todo'" key="todo" :size="13" />
          <svg
            v-else
            :key="t.status"
            width="13"
            height="13"
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
</template>

<style scoped lang="scss">
// Soft neon ground, opaque so the sticky card hides the thread scrolling
// under it.
.plan-card {
  width: 100%;
  margin: 0;
  padding: 5px 10px;
  border-radius: 12px;
  list-style: none;
  background: linear-gradient(
    135deg,
    color-mix(in srgb, var(--pink) 6%, var(--bg)),
    color-mix(in srgb, var(--violet) 6%, var(--bg))
  );
}

.plan-task {
  display: flex;
  align-items: center;
  gap: 6px;
  min-height: 21px;
  font-size: var(--fs-mono);
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
  width: 13px;
  height: 13px;
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

// Read by screen readers only — the mark carries it visually.
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
