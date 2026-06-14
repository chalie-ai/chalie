<script setup lang="ts">
/**
 * TaskDrawer — slide-out panel showing pending reminders and active subagents.
 *
 * Port source: frontend/interface/task_strip.js
 *
 * The trigger button lives in PresenceBar.vue (legacy index.html lines 122-127).
 * This component owns the scrim, drawer panel, and close button only.
 * Open/close state is driven by tasks.isOpen (tasks store).
 *
 * D4 fix: hint fires when the drawer opens with content, not on mount.
 *   Legacy task_strip.js lines 207-210: hint is rendered inside _render() which
 *   early-returns when totalCount === 0 — so the hint can only appear with content.
 *
 * D5 fix: watcher closes drawer when count drops to 0.
 *   Legacy task_strip.js lines 164-169: _closeDrawer() called when totalCount === 0.
 *
 * Key adaptation: the backend /chat/subagents/active only returns { sub_id }.
 * There is no agent_type or description field, so each subagent is rendered
 * with a generic "Working…" label alongside its sub_id.
 */
import { ref, computed, watch, onMounted, onBeforeUnmount } from 'vue';
import { storeToRefs } from 'pinia';
import { useTasksStore } from '../../stores/tasks';
import { scheduler } from '../../api/scheduler';
import { webPlatformAdapter } from '@chalie/shared';
import { relativeTime } from '../../utils/time';

const HINT_KEY = 'task_strip_hint_shown';
const POLL_INTERVAL_MS = 60_000;

// ── Store ──────────────────────────────────────────────────────────────────────

const tasks = useTasksStore();
const { reminders, subagents, totalCount, isOpen } = storeToRefs(tasks);

// ── Local state ───────────────────────────────────────────────────────────────

const hintShown = ref(webPlatformAdapter.getItem(HINT_KEY) === '1');
let pollTimer: ReturnType<typeof setInterval> | null = null;

// ── Computed ──────────────────────────────────────────────────────────────────

const hasReminders = computed(() => reminders.value.length > 0);
const hasSubagents = computed(() => subagents.value.size > 0);

/** Subagents as an array so v-for can iterate. */
const subagentList = computed(() => Array.from(subagents.value.values()));

/**
 * Whether the first-time hint should be shown.
 * D4: only show when the drawer is open AND there is content — mirrors legacy
 * task_strip.js where hint HTML is injected inside _render() after the
 * early-return guard at line 165 (totalCount === 0 → return).
 */
const showHint = computed(() => !hintShown.value && isOpen.value && totalCount.value > 0);

// ── Drawer DOM refs (for CSS transition choreography) ─────────────────────────

const drawerRef = ref<HTMLElement | null>(null);
const scrimRef = ref<HTMLElement | null>(null);

function openDrawerDom(): void {
  if (!scrimRef.value || !drawerRef.value) return;
  scrimRef.value.classList.remove('hidden');
  drawerRef.value.classList.remove('hidden');
  requestAnimationFrame(() => {
    drawerRef.value?.classList.add('open');
  });
  // D4: mark hint shown on first open-with-content.
  if (totalCount.value > 0 && !hintShown.value) {
    hintShown.value = true;
    webPlatformAdapter.setItem(HINT_KEY, '1');
  }
}

function closeDrawerDom(): void {
  const drawer = drawerRef.value;
  const scrim = scrimRef.value;
  if (!drawer) return;

  drawer.classList.remove('open');
  scrim?.classList.add('hidden');

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

// ── Watch store isOpen ─────────────────────────────────────────────────────────

watch(isOpen, (open) => {
  if (open) {
    openDrawerDom();
  } else {
    closeDrawerDom();
  }
});

// ── D5: Auto-close when count drops to 0 ─────────────────────────────────────
// Legacy task_strip.js lines 164-169: closes drawer when totalCount === 0.

watch(totalCount, (count) => {
  if (count === 0 && isOpen.value) {
    tasks.close();
  }
});

// ── Stop subagent ─────────────────────────────────────────────────────────────

async function stopSubagent(subId: string): Promise<void> {
  try {
    await scheduler.subagentStop(subId);
  } catch (err) {
    console.warn('[TaskDrawer] Subagent stop request failed:', err);
  }
}

// ── Keyboard ──────────────────────────────────────────────────────────────────

function handleKeydown(e: KeyboardEvent): void {
  if (e.key === 'Escape' && isOpen.value) {
    tasks.close();
  }
}

// ── Lifecycle ─────────────────────────────────────────────────────────────────

onMounted(() => {
  document.addEventListener('keydown', handleKeydown);

  // Initial load + periodic poll.
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
  <!-- Scrim -->
  <div
    id="taskDrawerScrim"
    ref="scrimRef"
    class="task-drawer__scrim hidden"
    aria-hidden="true"
    @click="tasks.close()"
  ></div>

  <!-- Slide-out panel -->
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
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
          <line x1="18" y1="6" x2="6" y2="18"></line>
          <line x1="6" y1="6" x2="18" y2="18"></line>
        </svg>
      </button>
    </div>

    <div id="taskDrawerList" class="task-drawer__list">
      <!-- Reminders section -->
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

      <!-- Divider between reminders and subagents -->
      <div
        v-if="hasReminders && hasSubagents"
        class="task-drawer__section-divider"
      ></div>

      <!-- Subagents section — backend only provides sub_id -->
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
            <svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor" aria-hidden="true">
              <rect x="1" y="1" width="10" height="10" rx="2"/>
            </svg>
          </button>
        </div>
      </template>

      <!-- First-time hint — shown on first open-with-content (D4). -->
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
// ── Scrim ──────────────────────────────────────────────────────────────────────

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

// ── Drawer panel ───────────────────────────────────────────────────────────────

.task-drawer {
  position: fixed;
  top: 0;
  right: 0;
  height: 100%;
  width: 320px;
  max-width: 90vw;
  background: var(--surface-elevated, var(--surface));
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

// ── List ───────────────────────────────────────────────────────────────────────

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
