<script setup lang="ts">
/**
 * Subagents dock — slide-out panel listing every subagent still running,
 * oldest first, from the agents store. Each row is its subagent's type icon
 * and the one-line summary of the call that started it — the type's name when
 * the call has none, never the tool name or number — and opens its transcript
 * in the delegate panel under the same title.
 *
 * Open/close is driven by session.subagentDockOpen.
 */
import { computed } from 'vue';
import type { DelegateRef } from '@chalie/shared';
import { useAgentsStore } from '../../stores/agents';
import { useSessionStore } from '../../stores/session';
import { subagentKind } from '../../utils/delegateChannel';
import SideDrawer from './SideDrawer.vue';

const session = useSessionStore();
const agents = useAgentsStore();

const rows = computed(() =>
  agents.running.map((agent) => {
    const kind = subagentKind(agent.type);
    return {
      delegate: { channel: agent.channel, turn_id: agent.turn_id },
      key: agent.agent_id,
      icon: kind.icon,
      title: agent.summary.trim() || kind.name,
    };
  }),
);

function openSubagent(delegate: DelegateRef, title: string): void {
  session.openDelegatePanel(delegate, title);
  session.closeSubagentDock();
}
</script>

<template>
  <SideDrawer
    :open="session.subagentDockOpen"
    title="Subagents"
    @close="session.closeSubagentDock()"
  >
    <p v-if="!rows.length" class="drawer-empty">No subagents running.</p>

    <button
      v-for="row in rows"
      :key="row.key"
      type="button"
      class="drawer-row"
      @click="openSubagent(row.delegate, row.title)"
    >
      <span class="drawer-row__top">
        <component :is="row.icon" class="drawer-row__icon" :size="16" aria-hidden="true" />
        <span class="drawer-row__label">{{ row.title }}</span>
        <span class="thread-activity-dot working" aria-hidden="true" />
      </span>
    </button>
  </SideDrawer>
</template>
