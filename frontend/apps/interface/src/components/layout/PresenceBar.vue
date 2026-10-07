<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue';
import { Brain, CalendarClock, Clock, Moon, Search, Sun } from '@lucide/vue';
import { useSessionStore } from '../../stores/session';
import { useTasksStore } from '../../stores/tasks';
import { ChalieMark, ConfigType, webPlatformAdapter, useTheme } from '@chalie/shared';
import { emit } from '../../composables/useEventBus';
import { useDockBusy } from '../../composables/useDockBusy';
import { useThreadActivity } from '../../utils/threadActivity';
import { hasDoneScheduled } from '../../utils/turnDom';

// Scheduler-dock activity cue: pink icon when any scheduled turn is
// "done" (settled unseen). Mirrors SchedulerDock.vue's bumpActivity pattern.
const hasSchedulerActivity = ref(hasDoneScheduled());

function onTurnStateChanged(): void {
  hasSchedulerActivity.value = hasDoneScheduled();
}
const session = useSessionStore();

// Replaces the retired `session.isSending` store getter — the mark's dot
// pulses while the main spine (no stable turn_id) has anything working.
const isSending = useDockBusy(() => null, () => ConfigType.USER);

const tasks = useTasksStore();

/** DOM-derived (no store) — see `utils/threadActivity.ts`. */
const threadActivity = useThreadActivity();
const totalCount = computed(() => threadActivity.value.length);

const { toggle, theme } = useTheme();

function handleThemeToggle(): void {
  toggle();
  emit('chalie:theme-changed', { theme: theme.value });
}

/** Settings button → open the Brain admin dashboard via the platform adapter. */
function handleSettings(): void {
  webPlatformAdapter.openBrain();
}

onMounted(() => {
  document.addEventListener('turn-state-changed', onTurnStateChanged);
});
onBeforeUnmount(() => {
  document.removeEventListener('turn-state-changed', onTurnStateChanged);
});
</script>

<template>
  <header class="presence-bar">
    <ChalieMark :size="28" :working="isSending" />
    <div class="presence-bar__right">
      <button
        id="searchBtn"
        class="btn-icon"
        aria-label="Search threads"
        title="Search threads (⌘K)"
        @click="session.openSearch()"
      >
        <Search :size="16" aria-hidden="true" />
      </button>
      <button
        id="schedulerDockBtn"
        class="btn-icon"
        :class="{ 'has-activity': hasSchedulerActivity }"
        aria-label="Schedules"
        title="Schedules"
        @click="session.openSchedulerDock()"
      >
        <CalendarClock :size="16" aria-hidden="true" />
      </button>
      <button
        id="taskDrawerBtn"
        class="btn-icon task-drawer-trigger"
        :class="{ 'has-activity': totalCount > 0 }"
        aria-label="Activity"
        title="Activity"
        @click="tasks.open()"
      >
        <Clock :size="16" aria-hidden="true" />
      </button>
      <button id="settingsBtn" class="btn-icon" aria-label="Settings" @click="handleSettings">
        <Brain :size="16" />
      </button>
      <button
        id="themeBtn"
        class="btn-icon"
        aria-label="Toggle light or dark theme"
        title="Toggle light / dark"
        @click="handleThemeToggle"
      >
        <Moon v-if="theme === 'dark'" :size="16" aria-hidden="true" />
        <Sun v-else :size="16" aria-hidden="true" />
      </button>
    </div>
  </header>
</template>

<style scoped lang="scss">
.task-drawer-trigger {
  position: relative;
  display: inline-flex;
  align-items: center;
  justify-content: center;
}

// Pink cue on the scheduler dock button when any scheduled turn is
// "done" (settled unseen). The CalendarClock SVG inherits this via
// currentColor, so setting it on the button cascades to the icon.
.has-activity {
  color: var(--pink-text);
}
</style>
