<script setup lang="ts">
/**
 * Activity drawer — a slide-out panel with two lists, each under its own
 * title and shown only when it has rows: "Threads", the forked threads whose
 * reply has settled unseen, and under it "Agents", every subagent still
 * running or waiting for a reply, oldest first, from the agents store.
 *
 * A subagent row is its type's icon and the one-line summary of the call that
 * started it — the type's name when the call has none, never the tool name or
 * number — then the pulsing dot while it runs, or the word "waiting" while it
 * waits for a reply; it opens its transcript in the delegate panel under that
 * title.
 *
 * The trigger button lives in PresenceBar.vue and shows only while either list
 * has rows; the slide-out shell (scrim, panel, close, transition choreography)
 * is SideDrawer. This component supplies the rows and wires them to
 * tasks.isOpen. The hint appears on first open-with-content, and the panel
 * auto-closes when the last row of both lists clears.
 */
import { computed, ref, watch } from 'vue';
import { storeToRefs } from 'pinia';
import type { DelegateRef } from '@chalie/shared';
import { useTasksStore } from '../../stores/tasks';
import { useSessionStore } from '../../stores/session';
import { useAgentsStore } from '../../stores/agents';
import { webPlatformAdapter } from '@chalie/shared';
import { useThreadActivity } from '../../utils/threadActivity';
import { subagentKind } from '../../utils/delegateChannel';
import SideDrawer from './SideDrawer.vue';

const HINT_KEY = 'task_strip_hint_shown';

// ── Store ──────────────────────────────────────────────────────────────────────

const tasks = useTasksStore();
const session = useSessionStore();
const agents = useAgentsStore();
const { isOpen } = storeToRefs(tasks);

/** DOM-derived (no store) — see `utils/threadActivity.ts`. */
const threadActivity = useThreadActivity();

const agentRows = computed(() =>
  agents.active.map((agent) => {
    const kind = subagentKind(agent.type);
    return {
      delegate: { channel: agent.channel, turn_id: agent.turn_id },
      key: agent.agent_id,
      icon: kind.icon,
      title: agent.summary.trim() || kind.name,
      waiting: agent.status === 'waiting',
    };
  }),
);

const totalCount = computed(() => threadActivity.value.length + agentRows.value.length);

/** Open a thread's slide-over from its Activity row, then close the drawer. */
function openThread(turnId: number, type: string): void {
  session.openThreadPanel(turnId, type);
  tasks.close();
}

/** Open a subagent's transcript from its Activity row, then close the drawer. */
function openSubagent(delegate: DelegateRef, title: string): void {
  session.openDelegatePanel(delegate, title);
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
    <!-- Forked threads whose reply has settled unseen (done). Clicking
         opens the thread's slide-over. The mockup's floating notifications
         live here. -->
    <template v-if="threadActivity.length">
      <h3 class="task-drawer__section">Threads</h3>
      <button
        v-for="ta in threadActivity"
        :key="`thread-${ta.turn_id}`"
        class="task-drawer__thread"
        @click="openThread(ta.turn_id, ta.type)"
      >
        <span class="task-drawer__thread-top">
          <span class="task-drawer__thread-label">{{ ta.label }}</span>
          <span class="thread-activity-dot" :class="ta.kind" aria-hidden="true" />
        </span>
        <span class="task-drawer__thread-snippet">{{ ta.snippet }}</span>
      </button>
    </template>

    <!-- Subagents still running or waiting for a reply. Clicking opens the
         subagent's transcript. -->
    <template v-if="agentRows.length">
      <h3 class="task-drawer__section">Agents</h3>
      <button
        v-for="row in agentRows"
        :key="row.key"
        type="button"
        class="drawer-row"
        @click="openSubagent(row.delegate, row.title)"
      >
        <span class="drawer-row__top">
          <component :is="row.icon" class="drawer-row__icon" :size="16" aria-hidden="true" />
          <span class="drawer-row__label">{{ row.title }}</span>
          <span v-if="row.waiting" class="task-drawer__status">waiting</span>
          <span v-else class="thread-activity-dot working" aria-hidden="true" />
        </span>
      </button>
    </template>

    <!-- First-time hint — shown on first open-with-content. -->
    <div v-if="showHint" class="task-drawer__hint">I'll show what I'm working on here.</div>
  </SideDrawer>
</template>

<style scoped lang="scss">
// ── Section title ─────────────────────────────────────────────────────────────
// A mono label over each list, styled like the spine's day marker.

.task-drawer__section {
  margin: 0;
  padding: 12px 16px 4px;
  font-family: var(--font-mono);
  font-size: var(--fs-mono);
  font-weight: 400;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--muted);
}

// ── Agent status word ─────────────────────────────────────────────────────────
// Stands in for the working dot while an agent waits for a reply, in the
// section title's mono type.

.task-drawer__status {
  font-family: var(--font-mono);
  font-size: var(--fs-mono);
  color: var(--muted);
}

// ── Thread-activity row ──────────────────────────────────────────────────────────
// Forked threads whose reply has settled unseen, folded out of the mockup's
// floating notifications. The steady activity dot marks them done.

.task-drawer__thread {
  display: flex;
  flex-direction: column;
  gap: 4px;
  width: 100%;
  text-align: left;
  background: transparent;
  padding: 10px 16px;
  cursor: pointer;
  transition:
    background-color var(--dur-1) var(--ease-out),
    translate var(--dur-1) var(--ease-out);

  &:hover {
    background: var(--cell);
  }

  &:active {
    translate: 0 1px;
  }
}

.task-drawer__thread-top {
  display: flex;
  align-items: center;
  gap: 8px;
}

.task-drawer__thread-label {
  flex: 1;
  min-width: 0;
  font-size: var(--fs-body);
  font-weight: 600;
  color: var(--text);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.task-drawer__thread-snippet {
  font-size: var(--fs-body);
  line-height: 1.45;
  color: var(--muted);
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.task-drawer__hint {
  font-size: var(--fs-body);
  color: var(--muted);
  padding: 8px 16px 4px;
  font-style: italic;
}
</style>
