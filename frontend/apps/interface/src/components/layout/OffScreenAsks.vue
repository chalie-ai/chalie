<!-- One dock row per open permission or sign-in code ask no live line on
     screen shows: a scheduled task's, one from a thread whose panel is closed,
     or a subagent's raised after the turn that started it ended. Deny and
     Allow, or a typed code, answer the ask right here; Open shows its turn in
     the thread panel. A browser hand-off gets a row while the browser panel is
     closed, or while it is open on behalf of another turn; its Open shows the
     tab in the browser panel, where Done answers it. Every open question is
     answered on the one card under the rows, nearest the message box. -->
<script setup lang="ts">
import { computed } from 'vue';
import {
  AskRow,
  QuestionCard,
  useAsksStore,
  useWaitClock,
  type InlineAsk,
} from '@chalie/shared';
import { useSessionStore } from '../../stores/session';
import { useBrowserStore } from '../../stores/browser';

const session = useSessionStore();
const asks = useAsksStore();
const browser = useBrowserStore();
const browserAway = computed(() =>
  browser.awayFrom({ turnId: session.panelThreadId, type: session.panelType }),
);

const now = useWaitClock(() => asks.offScreen.length + browserAway.value.length + asks.questions.length);

function openTurn(ask: InlineAsk): void {
  if (ask.data.origin) session.openThreadPanel(ask.data.origin.turn_id, ask.data.origin.type);
}
</script>

<template>
  <!-- Held through the conversation's first load: until its turns render, an
       ask their live lines are about to claim would flash here first. -->
  <div class="offscreen-asks" aria-live="polite">
    <AskRow
      v-for="ask in session.historyLoading ? [] : asks.offScreen"
      :key="ask.ask_id"
      :ask="ask"
      :now="now"
      @open="openTurn(ask)"
    />
    <AskRow
      v-for="ask in session.historyLoading ? [] : browserAway"
      :key="ask.ask_id"
      :ask="ask"
      :now="now"
      @open="browser.openPanel(ask.data.tab_id)"
    />
    <QuestionCard :questions="session.historyLoading ? [] : asks.questions" :now="now" @open="openTurn" />
  </div>
</template>
