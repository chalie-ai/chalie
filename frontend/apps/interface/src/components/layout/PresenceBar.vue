<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue';
import { Brain, CalendarClock, Clock, Moon, Search, Sun } from '@lucide/vue';
import { useSessionStore } from '../../stores/session';
import { useTasksStore } from '../../stores/tasks';
import { ConfigType, platform, useTheme } from '@chalie/shared';
import { emit } from '../../composables/useEventBus';
import { useDockBusy } from '../../composables/useDockBusy';
import { useThreadActivity } from '../../utils/threadActivity';
import { hasDoneScheduled } from '../../utils/turnDom';

// D16 — scheduler-dock activity cue: aqua icon when any scheduled turn is
// "done" (settled unseen). Mirrors SchedulerDock.vue's bumpActivity pattern.
const hasSchedulerActivity = ref(hasDoneScheduled());

function onTurnStateChanged(): void {
  hasSchedulerActivity.value = hasDoneScheduled();
}
const session = useSessionStore();

// D3: replaces the retired `session.isSending` store getter — the logo
// breathes while the main spine (no stable turn_id) has anything working.
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
  platform.openBrain();
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
    <img
      class="presence-logo"
      :class="{ 'presence-logo--active': isSending }"
      src="/icons/icon.png"
      alt="Chalie"
    />
    <div class="presence-bar__right">
      <button
        id="searchBtn"
        class="btn-icon"
        aria-label="Search threads"
        title="Search threads (⌘K)"
        @click="session.openSearch()"
      >
        <Search :size="18" aria-hidden="true" />
      </button>
      <button
        id="schedulerDockBtn"
        class="btn-icon"
        :class="{ 'has-activity': hasSchedulerActivity }"
        aria-label="Schedules"
        title="Schedules"
        @click="session.openSchedulerDock()"
      >
        <CalendarClock :size="18" aria-hidden="true" />
      </button>
      <button
        id="taskDrawerBtn"
        class="btn-icon task-drawer-trigger"
        :class="{ 'has-activity': totalCount > 0 }"
        aria-label="Activity"
        title="Activity"
        @click="tasks.open()"
      >
        <Clock :size="18" aria-hidden="true" />
      </button>
      <button id="settingsBtn" class="btn-icon" aria-label="Settings" @click="handleSettings">
        <Brain :size="18" />
      </button>
      <button
        id="themeBtn"
        class="btn-icon"
        aria-label="Toggle light or dark theme"
        title="Toggle light / dark"
        @click="handleThemeToggle"
      >
        <Moon v-if="theme === 'dark'" :size="18" aria-hidden="true" />
        <Sun v-else :size="18" aria-hidden="true" />
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

// Chalie logo — the bar's left mark. Static at rest; a slow, slight opacity
// breathe (0.7 → 1) while a turn is in flight, in sync with the in-feed
// "thinking…" anchor. The gradient mark reads on both theme scrims (Rule 7).
.presence-logo {
  height: 30px;
  width: auto;
  display: block;
  user-select: none;
  -webkit-user-drag: none;
}

.presence-logo--active {
  animation: presence-logo-breathe 2.5s ease-in-out infinite;
}

@keyframes presence-logo-breathe {
  0%,
  100% {
    opacity: 0.7;
  }
  50% {
    opacity: 1;
  }
}

// D16 — aqua cue on the scheduler dock button when any scheduled turn is
// "done" (settled unseen). The CalendarClock SVG inherits this via
// currentColor, so setting it on the button cascades to the icon.
.has-activity {
  color: var(--cyan);
}
</style>
