<script setup lang="ts">
/**
 * Activity drawer — a slide-out panel that lists forked threads whose reply
 * has settled unseen, one row each, no section split.
 *
 * The trigger button lives in PresenceBar.vue; the slide-out shell (scrim,
 * panel, close, transition choreography) is SideDrawer. This component supplies
 * the rows and wires them to tasks.isOpen. The hint appears on first
 * open-with-content; the panel auto-closes when the last item clears.
 */
import { computed, ref, watch } from 'vue';
import { storeToRefs } from 'pinia';
import { useTasksStore } from '../../stores/tasks';
import { useSessionStore } from '../../stores/session';
import { webPlatformAdapter } from '@chalie/shared';
import { useThreadActivity } from '../../utils/threadActivity';
import SideDrawer from './SideDrawer.vue';

const HINT_KEY = 'task_strip_hint_shown';

// ── Store ──────────────────────────────────────────────────────────────────────

const tasks = useTasksStore();
const session = useSessionStore();
const { isOpen } = storeToRefs(tasks);

/** DOM-derived (no store) — see `utils/threadActivity.ts`. */
const threadActivity = useThreadActivity();
const totalCount = computed(() => threadActivity.value.length);

/** Open a thread's slide-over from its Activity row, then close the drawer. */
function openThread(turnId: number, type: string): void {
  session.openThreadPanel(turnId, type);
  tasks.close();
}

// ── Local state ───────────────────────────────────────────────────────────────

const hintShown = ref(webPlatformAdapter.getItem(HINT_KEY) === '1');

// ── Computed ──────────────────────────────────────────────────────────────────

/** Show the first-time hint only when the drawer is open with content. */
const showHint = computed(() => !hintShown.value && isOpen.value && totalCount.value > 0);

// ── Watch store isOpen ─────────────────────────────────────────────────────────

watch(isOpen, (open) => {
  if (open) {
    // First-time hint: mark shown on the first open that actually has content.
    if (totalCount.value > 0 && !hintShown.value) {
      hintShown.value = true;
      webPlatformAdapter.setItem(HINT_KEY, '1');
    }
  }
});

// ── Auto-close when the last item clears ─────────────────────────────────────

watch(totalCount, (count) => {
  if (count === 0 && isOpen.value) {
    tasks.close();
  }
});
</script>

<template>
  <SideDrawer :open="isOpen" title="Activity" @close="tasks.close()">
    <!-- Forked threads whose reply has settled unseen (done, blue). Clicking
         opens the thread's slide-over. The mockup's floating notifications
         live here. -->
    <template v-if="threadActivity.length">
      <button
        v-for="ta in threadActivity"
        :key="`thread-${ta.turn_id}`"
        class="task-drawer__thread"
        :class="`task-drawer__thread--${ta.kind}`"
        @click="openThread(ta.turn_id, ta.type)"
      >
        <span class="task-drawer__thread-top">
          <span class="task-drawer__thread-label">{{ ta.label }}</span>
          <span class="thread-activity-dot" :class="ta.kind" aria-hidden="true" />
        </span>
        <span class="task-drawer__thread-snippet">{{ ta.snippet }}</span>
      </button>
    </template>

    <!-- First-time hint — shown on first open-with-content. -->
    <div v-if="showHint" class="task-drawer__hint">I'll show what I'm working on here.</div>
  </SideDrawer>
</template>

<style scoped lang="scss">
// ── Thread-activity row ──────────────────────────────────────────────────────────
// Forked threads whose reply has settled unseen, folded out of the mockup's
// floating notifications. A cyan left accent stripe marks them done.

.task-drawer__thread {
  display: flex;
  flex-direction: column;
  gap: 4px;
  width: 100%;
  text-align: left;
  background: transparent;
  border: none;
  border-left: 2px solid transparent;
  padding: 10px 16px;
  cursor: pointer;

  &:hover {
    background: var(--surface-hover, rgba(128, 128, 128, 0.06));
  }
}

.task-drawer__thread--done {
  border-left-color: var(--cyan);
}

.task-drawer__thread-top {
  display: flex;
  align-items: center;
  gap: 8px;
}

.task-drawer__thread-label {
  flex: 1;
  min-width: 0;
  font-size: 13px;
  font-weight: 600;
  color: var(--text-primary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.task-drawer__thread-snippet {
  font-size: 11.5px;
  line-height: 1.45;
  color: var(--text-secondary);
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.task-drawer__hint {
  font-size: 12px;
  color: var(--text-secondary);
  padding: 8px 16px 4px;
  font-style: italic;
}
</style>
