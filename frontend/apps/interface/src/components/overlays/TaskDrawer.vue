<script setup lang="ts">
/**
 * Activity drawer — a slide-out panel that merges everything Chalie is doing or
 * is about to do into one list: scheduled items (the pending scheduler list) and
 * active delegates (backgrounded tool calls), one row each, no section split.
 *
 * A delegate row shows the model's summary of what it's doing (bold title), the
 * delegate's tool name (subtitle), and a foot row with a live elapsed timer on
 * the left and a stop control on the right. Stop flips the delegate's cancel
 * event server-side (DELETE /api/subagents/<id>); the row shows "Stopping…"
 * until the subagent_end push removes it.
 *
 * The trigger button lives in PresenceBar.vue. This component owns the scrim,
 * panel, and close button only; open/close state is driven by tasks.isOpen.
 * The hint appears on first open-with-content; the panel auto-closes when the
 * last item clears.
 */
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue';
import { Square, X } from '@lucide/vue';
import { storeToRefs } from 'pinia';
import { useTasksStore } from '../../stores/tasks';
import { useSessionStore } from '../../stores/session';
import type { ActiveSubagent } from '../../api/scheduler';
import { scheduler } from '../../api/scheduler';
import { webPlatformAdapter } from '@chalie/shared';
import { elapsedSince } from '../../utils/time';
import { useThreadActivity } from '../../utils/threadActivity';

const HINT_KEY = 'task_strip_hint_shown';
const POLL_INTERVAL_MS = 60_000;
const TICK_INTERVAL_MS = 1_000;

// ── Store ──────────────────────────────────────────────────────────────────────

const tasks = useTasksStore();
const session = useSessionStore();
const { subagents, isOpen } = storeToRefs(tasks);

/** DOM-derived (no store) — see `utils/threadActivity.ts`. */
const threadActivity = useThreadActivity();
const totalCount = computed(() => subagents.value.size + threadActivity.value.length);

/** Open a thread's slide-over from its Activity row, then close the drawer. */
function openThread(turnId: number): void {
  session.openThreadPanel(turnId);
  tasks.close();
}

// ── Local state ───────────────────────────────────────────────────────────────

const hintShown = ref(webPlatformAdapter.getItem(HINT_KEY) === '1');
let pollTimer: ReturnType<typeof setInterval> | null = null;

/** Live wall-clock driving every delegate's elapsed timer; ticks only while open. */
const nowMs = ref(Date.now());
let tickTimer: ReturnType<typeof setInterval> | null = null;

/**
 * Delegates whose stop was requested — rendered as "Stopping…" until the
 * subagent_end push removes the row. Per-open state, cleared on close.
 */
const stopping = ref(new Set<string>());

// ── Computed ──────────────────────────────────────────────────────────────────

const hasSubagents = computed(() => subagents.value.size > 0);

/** Delegates as an array so v-for can iterate. */
const subagentList = computed(() => Array.from(subagents.value.values()));

/** Show the first-time hint only when the drawer is open with content. */
const showHint = computed(() => !hintShown.value && isOpen.value && totalCount.value > 0);

/** Row title: the model's summary, falling back to the tool name. */
function delegateTitle(sa: ActiveSubagent): string {
  return sa.summary || sa.tool_name;
}

function elapsed(sa: ActiveSubagent): string {
  return elapsedSince(sa.started_at, nowMs.value);
}

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

// ── Elapsed-timer ticker (open-only) ──────────────────────────────────────────

function startTick(): void {
  if (tickTimer !== null) return;
  nowMs.value = Date.now();
  tickTimer = setInterval(() => {
    nowMs.value = Date.now();
  }, TICK_INTERVAL_MS);
}

function stopTick(): void {
  if (tickTimer === null) return;
  clearInterval(tickTimer);
  tickTimer = null;
}

// ── Watch store isOpen ─────────────────────────────────────────────────────────

watch(isOpen, (open) => {
  if (open) {
    openDrawerDom();
    startTick();
  } else {
    closeDrawerDom();
    stopTick();
    stopping.value.clear();
  }
});

// ── Auto-close when the last item clears ─────────────────────────────────────

watch(totalCount, (count) => {
  if (count === 0 && isOpen.value) {
    tasks.close();
  }
});

// ── Stop delegate ─────────────────────────────────────────────────────────────

async function stopSubagent(subId: string): Promise<void> {
  stopping.value.add(subId);
  try {
    await scheduler.subagentStop(subId);
  } catch (err) {
    console.warn('[ActivityDrawer] delegate stop request failed:', err);
    stopping.value.delete(subId); // surface the failure: let the user retry
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
  void tasks.loadActiveTasks();
  pollTimer = setInterval(async () => {
    await tasks.loadActiveTasks();
  }, POLL_INTERVAL_MS);
});

onBeforeUnmount(() => {
  document.removeEventListener('keydown', handleKeydown);
  if (pollTimer !== null) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
  stopTick();
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
    aria-label="Activity"
  >
    <div class="task-drawer__header">
      <h2 class="task-drawer__title">Activity</h2>
      <button
        id="taskDrawerClose"
        class="btn-icon task-drawer__close"
        aria-label="Close activity panel"
        @click="tasks.close()"
      >
        <X :size="16" aria-hidden="true" />
      </button>
    </div>

    <div id="taskDrawerList" class="task-drawer__list">
      <!-- Live forked threads — reply streaming (pink) or settled-unseen (blue).
           Clicking opens the thread's slide-over. The mockup's floating
           notifications live here. -->
      <template v-if="threadActivity.length">
        <button
          v-for="ta in threadActivity"
          :key="`thread-${ta.turn_id}`"
          class="task-drawer__thread"
          :class="`task-drawer__thread--${ta.kind}`"
          @click="openThread(ta.turn_id)"
        >
          <span class="task-drawer__thread-top">
            <span class="task-drawer__thread-label">{{ ta.label }}</span>
            <span class="task-drawer__thread-dot" :class="ta.kind" aria-hidden="true" />
          </span>
          <span class="task-drawer__thread-snippet">{{ ta.snippet }}</span>
        </button>
      </template>

      <!-- Active delegates — what Chalie is doing right now -->
      <template v-if="hasSubagents">
        <div
          v-for="sa in subagentList"
          :key="sa.sub_id"
          class="task-drawer__delegate"
          :class="{ 'is-stopping': stopping.has(sa.sub_id) }"
        >
          <div class="task-drawer__delegate-title">{{ delegateTitle(sa) }}</div>
          <div class="task-drawer__delegate-type">{{ sa.tool_name }}</div>
          <div class="task-drawer__delegate-foot">
            <span class="task-drawer__timer">{{ elapsed(sa) }}</span>
            <button
              v-if="!stopping.has(sa.sub_id)"
              class="task-drawer__stop"
              :aria-label="`Stop ${sa.tool_name}`"
              @click="stopSubagent(sa.sub_id)"
            >
              <Square :size="11" fill="currentColor" aria-hidden="true" />
              <span>Stop</span>
            </button>
            <span v-else class="task-drawer__stopping">Stopping&hellip;</span>
          </div>
        </div>
      </template>

      <!-- First-time hint — shown on first open-with-content. -->
      <div v-if="showHint" class="task-drawer__hint">I'll show what I'm working on here.</div>
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

// ── List ───────────────────────────────────────────────────────────────────────

.task-drawer__list {
  flex: 1;
  overflow-y: auto;
  padding: 8px 0;
}

// ── Thread-activity row ──────────────────────────────────────────────────────────
// Live forked threads, folded out of the mockup's floating notifications. A left
// accent stripe (pink while working, blue once done) tells the two apart.

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

.task-drawer__thread--working {
  border-left-color: var(--status-main);
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

// ── Delegate row ───────────────────────────────────────────────────────────────

.task-drawer__delegate {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 10px 16px;

  &:hover {
    background: var(--surface-hover, rgba(128, 128, 128, 0.06));
  }

  &.is-stopping {
    opacity: 0.55;
  }
}

// Bold title: the model's summary, capped at two lines.
.task-drawer__delegate-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--text-primary);
  line-height: 1.35;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.task-drawer__delegate-type {
  font-size: 11px;
  color: var(--text-secondary);
  line-height: 1.3;
}

.task-drawer__delegate-foot {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  margin-top: 6px;
}

.task-drawer__timer {
  font-size: 12px;
  color: var(--text-secondary);
  font-variant-numeric: tabular-nums;
}

.task-drawer__stop {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-size: 11px;
  font-weight: 600;
  color: var(--text-secondary);
  background: var(--surface, var(--bg));
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 3px 8px;
  cursor: pointer;
  transition:
    color 0.15s ease,
    border-color 0.15s ease;

  &:hover {
    color: var(--error, #e55);
    border-color: var(--error, #e55);
  }
}

.task-drawer__stopping {
  font-size: 11px;
  font-style: italic;
  color: var(--text-secondary);
}

.task-drawer__hint {
  font-size: 12px;
  color: var(--text-secondary);
  padding: 8px 16px 4px;
  font-style: italic;
}
</style>
