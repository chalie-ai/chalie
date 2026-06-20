<script setup lang="ts">
/**
 * TaskDrawer — slide-out panel showing pending reminders and active subagents.
 *
 * The trigger button lives in PresenceBar.vue; this component owns the scrim,
 * panel, and close button only. Open/close state is driven by tasks.isOpen.
 *
 * The backend /chat/subagents/active returns only { sub_id } (no agent_type or
 * description), so each subagent renders as a generic "Working…" label.
 */
import { ref, computed, watch, onMounted, onBeforeUnmount } from 'vue';
import { Square, X } from '@lucide/vue';
import { storeToRefs } from 'pinia';
import { useTasksStore } from '../../stores/tasks';
import { scheduler } from '../../api/scheduler';
import { webPlatformAdapter } from '@chalie/shared';
import { relativeTime } from '../../utils/time';

const HINT_KEY = 'task_strip_hint_shown';
const POLL_INTERVAL_MS = 60_000;

const tasks = useTasksStore();
const { reminders, subagents, totalCount, isOpen } = storeToRefs(tasks);

const hintShown = ref(webPlatformAdapter.getItem(HINT_KEY) === '1');
let pollTimer: ReturnType<typeof setInterval> | null = null;

const hasReminders = computed(() => reminders.value.length > 0);
const hasSubagents = computed(() => subagents.value.size > 0);

const subagentList = computed(() => Array.from(subagents.value.values()));

/** First-time hint: only when the drawer is open AND has content. */
const showHint = computed(() => !hintShown.value && isOpen.value && totalCount.value > 0);

const drawerRef = ref<HTMLElement | null>(null);
const scrimRef = ref<HTMLElement | null>(null);

function openDrawerDom(): void {
  if (!scrimRef.value || !drawerRef.value) return;
  scrimRef.value.classList.remove('hidden');
  drawerRef.value.classList.remove('hidden');
  requestAnimationFrame(() => {
    drawerRef.value?.classList.add('open');
  });
  // Mark hint shown on first open-with-content.
  if (totalCount.value > 0 && !hintShown.value) {
    hintShown.value = true;
    webPlatformAdapter.setItem(HINT_KEY, '1');
  }
}

function closeDrawerDom(): void {
  const drawer = drawerRef.value;
  if (!drawer) return;

  drawer.classList.remove('open');
  scrimRef.value?.classList.add('hidden');

  drawer.addEventListener(
    'transitionend',
    () => {
      if (!drawer.classList.contains('open')) {
        drawer.classList.add('hidden');
      }
    },
    { once: true },
  );
}

watch(isOpen, (open) => {
  if (open) {
    openDrawerDom();
  } else {
    closeDrawerDom();
  }
});

// Auto-close when the count drops to 0.
watch(totalCount, (count) => {
  if (count === 0 && isOpen.value) {
    tasks.close();
  }
});

async function stopSubagent(subId: string): Promise<void> {
  try {
    await scheduler.subagentStop(subId);
  } catch (err) {
    console.warn('[TaskDrawer] Subagent stop request failed:', err);
  }
}

function handleKeydown(e: KeyboardEvent): void {
  if (e.key === 'Escape' && isOpen.value) {
    tasks.close();
  }
}

onMounted(() => {
  document.addEventListener('keydown', handleKeydown);

  void tasks.loadActiveTasks();
  pollTimer = setInterval(() => { void tasks.loadActiveTasks(); }, POLL_INTERVAL_MS);
});

onBeforeUnmount(() => {
  document.removeEventListener('keydown', handleKeydown);
  if (pollTimer !== null) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
});
</script>

<template>
  <div
    id="taskDrawerScrim"
    ref="scrimRef"
    class="task-drawer__scrim hidden"
    aria-hidden="true"
    @click="tasks.close()"
  ></div>

  <aside
    id="taskDrawer"
    ref="drawerRef"
    class="task-drawer hidden"
    role="complementary"
    aria-label="Active tasks"
  >
    <div class="task-drawer__header">
      <h2 class="task-drawer__title">Tasks</h2>
      <button
        id="taskDrawerClose"
        class="btn-icon task-drawer__close"
        aria-label="Close tasks panel"
        @click="tasks.close()"
      >
        <X :size="16" aria-hidden="true" />
      </button>
    </div>

    <div id="taskDrawerList" class="task-drawer__list">
      <template v-if="hasReminders">
        <div
          v-for="r in reminders"
          :key="r.id"
          class="task-drawer__item task-drawer__item--reminder"
        >
          <span class="task-drawer__msg">{{ r.message }}</span>
          <span v-if="r.due_at" class="task-drawer__due">{{ relativeTime(r.due_at) }}</span>
        </div>
      </template>

      <div
        v-if="hasReminders && hasSubagents"
        class="task-drawer__section-divider"
      ></div>

      <template v-if="hasSubagents">
        <div
          v-for="sa in subagentList"
          :key="sa.sub_id"
          class="task-drawer__item task-drawer__item--subagent"
        >
          <div class="task-drawer__subagent-row">
            <span class="task-drawer__msg">Working&hellip;</span>
            <span class="task-drawer__badge" :title="sa.sub_id">{{ sa.sub_id }}</span>
          </div>
          <button
            class="task-drawer__stop-btn btn-icon"
            :aria-label="`Stop subagent ${sa.sub_id}`"
            @click="stopSubagent(sa.sub_id)"
          >
            <Square :size="12" fill="currentColor" aria-hidden="true" />
          </button>
        </div>
      </template>

      <div
        v-if="showHint"
        class="task-drawer__hint"
      >
        I'll show what I'm working on here.
      </div>
    </div>
  </aside>
</template>

<style scoped lang="scss">
.task-drawer__scrim {
  position: fixed;
  inset: 0;
  background: var(--overlay-scrim, rgba(0, 0, 0, 0.35));
  z-index: 199;
  opacity: 1;
  transition: opacity 0.2s ease;

  &.hidden {
    display: none;
  }
}

.task-drawer {
  position: fixed;
  top: 0;
  right: 0;
  height: 100%;
  width: 320px;
  max-width: 90vw;
  background: var(--bg-2);
  border-left: 1px solid var(--border);
  box-shadow: -4px 0 24px var(--shadow, rgba(0, 0, 0, 0.15));
  z-index: 200;
  display: flex;
  flex-direction: column;
  transform: translateX(100%);
  transition: transform 0.25s ease;
  overflow: hidden;

  &.open {
    transform: translateX(0);
  }

  &.hidden {
    display: none;
  }
}

.task-drawer__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 16px 12px;
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
}

.task-drawer__title {
  margin: 0;
  font-size: 15px;
  font-weight: 600;
  color: var(--text-primary);
}

.task-drawer__close {
  color: var(--text-secondary);

  &:hover {
    color: var(--text-primary);
  }
}

.task-drawer__list {
  flex: 1;
  overflow-y: auto;
  padding: 8px 0;
}

.task-drawer__item {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 8px;
  padding: 10px 16px;

  &:hover {
    background: var(--surface-hover, rgba(128, 128, 128, 0.06));
  }
}

.task-drawer__item--reminder {
  flex-direction: column;
  gap: 2px;
}

.task-drawer__item--subagent {
  flex-direction: column;
  gap: 4px;
}

.task-drawer__subagent-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  width: 100%;
}

.task-drawer__msg {
  font-size: 13px;
  color: var(--text-primary);
  line-height: 1.4;
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.task-drawer__due {
  font-size: 11px;
  color: var(--text-secondary);
  font-variant-numeric: tabular-nums;
}

.task-drawer__badge {
  font-size: 10px;
  font-weight: 600;
  color: var(--text-secondary);
  background: var(--surface, var(--bg));
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 1px 5px;
  flex-shrink: 0;
  max-width: 100px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.task-drawer__stop-btn {
  color: var(--text-secondary);
  flex-shrink: 0;
  padding: 4px;

  &:hover {
    color: var(--error, #e55);
  }
}

.task-drawer__section-divider {
  height: 1px;
  background: var(--border);
  margin: 8px 16px;
}

.task-drawer__hint {
  font-size: 12px;
  color: var(--text-secondary);
  padding: 8px 16px 4px;
  font-style: italic;
}
</style>
