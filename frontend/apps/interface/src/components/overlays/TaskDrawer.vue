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
 * The trigger button lives in PresenceBar.vue; the slide-out shell (scrim,
 * panel, close, transition choreography) is SideDrawer. This component supplies
 * the rows and wires them to tasks.isOpen. The hint appears on first
 * open-with-content; the panel auto-closes when the last item clears.
 */
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue';
import { Square } from '@lucide/vue';
import { storeToRefs } from 'pinia';
import { useTasksStore } from '../../stores/tasks';
import { useSessionStore } from '../../stores/session';
import type { ActiveSubagent } from '../../api/scheduler';
import { scheduler } from '../../api/scheduler';
import { webPlatformAdapter } from '@chalie/shared';
import { elapsedSince } from '../../utils/time';
import { useThreadActivity } from '../../utils/threadActivity';
import SideDrawer from './SideDrawer.vue';

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
function openThread(turnId: number, type: string): void {
  session.openThreadPanel(turnId, type);
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
    // First-time hint: mark shown on the first open that actually has content.
    if (totalCount.value > 0 && !hintShown.value) {
      hintShown.value = true;
      webPlatformAdapter.setItem(HINT_KEY, '1');
    }
    startTick();
  } else {
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

// ── Lifecycle ─────────────────────────────────────────────────────────────────

onMounted(() => {
  void tasks.loadActiveTasks();
  pollTimer = setInterval(async () => {
    await tasks.loadActiveTasks();
  }, POLL_INTERVAL_MS);
});

onBeforeUnmount(() => {
  if (pollTimer !== null) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
  stopTick();
});
</script>

<template>
  <SideDrawer :open="isOpen" title="Activity" @close="tasks.close()">
    <!-- Live forked threads — reply streaming (pink) or settled-unseen (blue).
         Clicking opens the thread's slide-over. The mockup's floating
         notifications live here. -->
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
  </SideDrawer>
</template>

<style scoped lang="scss">
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
