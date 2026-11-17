<script setup lang="ts">
/**
 * Subagents dock — slide-out panel listing every subagent still running in a
 * turn on screen, in the order first seen. Each row is the one-line summary of
 * the call that started it (its subagent's name when the call has none) over
 * that subagent's name, and opens its transcript in the delegate panel.
 *
 * Open/close is driven by session.subagentDockOpen.
 */
import { computed } from 'vue';
import type { DelegateRef } from '@chalie/shared';
import { useSessionStore } from '../../stores/session';
import { delegateName } from '../../utils/delegateChannel';
import { runningSubagents } from '../../utils/liveActTrail';
import SideDrawer from './SideDrawer.vue';

const session = useSessionStore();

const rows = computed(() =>
  runningSubagents().map(({ delegate, summary }) => ({
    delegate,
    key: `${delegate.channel}#${delegate.turn_id}`,
    title: summary.trim() || delegateName(delegate.channel),
  })),
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
        <span class="drawer-row__label">{{ row.title }}</span>
        <span class="thread-activity-dot working" aria-hidden="true" />
      </span>
      <span class="drawer-row__sub">{{ delegateName(row.delegate.channel) }}</span>
    </button>
  </SideDrawer>
</template>
