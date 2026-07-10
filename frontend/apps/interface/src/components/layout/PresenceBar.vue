<script setup lang="ts">
import { Brain, CalendarClock, Clock, Moon, Search, Sun } from '@lucide/vue';
import { storeToRefs } from 'pinia';
import { useSessionStore } from '../../stores/session';
import { useTasksStore } from '../../stores/tasks';
import { ConfigType, platform, useTheme } from '@chalie/shared';
import { emit } from '../../composables/useEventBus';
import { useDockBusy } from '../../composables/useDockBusy';

const session = useSessionStore();

// D3: replaces the retired `session.isSending` store getter — the logo
// breathes while the main spine (no stable turn_id) has anything working.
const isSending = useDockBusy(() => null, () => ConfigType.USER);

const tasks = useTasksStore();
const { totalCount } = storeToRefs(tasks);

const { toggle, theme } = useTheme();

function handleThemeToggle(): void {
  toggle();
  emit('chalie:theme-changed', { theme: theme.value });
}

/** Settings button → open the Brain admin dashboard via the platform adapter. */
function handleSettings(): void {
  platform.openBrain();
}
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
        aria-label="Schedules"
        title="Schedules"
        @click="session.openSchedulerDock()"
      >
        <CalendarClock :size="18" aria-hidden="true" />
      </button>
      <button
        v-if="totalCount > 0"
        id="taskDrawerBtn"
        class="btn-icon task-drawer-trigger"
        aria-label="Activity"
        title="Activity"
        @click="tasks.open()"
      >
        <Clock :size="18" aria-hidden="true" />
        <span class="task-trigger__badge">{{ totalCount }}</span>
      </button>
      <button id="settingsBtn" class="btn-icon" aria-label="Settings" @click="handleSettings">
        <Brain :size="18" />
      </button>
      <button
        class="theme-toggle"
        :aria-label="'Toggle light or dark theme'"
        title="Toggle light / dark"
        @click="handleThemeToggle"
      >
        <span class="theme-toggle__track" aria-hidden="true">
          <Sun />
          <Moon />
        </span>
        <span class="theme-toggle__thumb" aria-hidden="true"></span>
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

.task-trigger__badge {
  position: absolute;
  top: -4px;
  right: -6px;
  min-width: 16px;
  height: 16px;
  padding: 0 4px;
  border-radius: 8px;
  background: var(--accent-primary);
  color: #fff;
  font-size: 10px;
  font-weight: 600;
  line-height: 16px;
  text-align: center;
  pointer-events: none;
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

.theme-toggle {
  position: relative;
  width: 56px;
  height: 30px;
  border-radius: var(--radius-full);
  background: var(--bg-input);
  border: 1px solid var(--border-subtle);
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  padding: 3px;
  transition:
    background 220ms ease,
    border-color 220ms ease;

  &:hover {
    border-color: var(--border-strong);
  }

  &:focus-visible {
    outline: 1.5px solid color-mix(in oklab, var(--violet) 45%, transparent);
    outline-offset: 2px;
  }
}

.theme-toggle__track {
  position: relative;
  width: 100%;
  height: 100%;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 6px;
  color: var(--text-tertiary);

  svg {
    width: 12px;
    height: 12px;
    stroke-width: 2;
  }
}

.theme-toggle__thumb {
  position: absolute;
  top: 2px;
  left: 2px;
  width: 24px;
  height: 24px;
  border-radius: 50%;
  background: linear-gradient(
    135deg,
    var(--violet),
    color-mix(in oklab, var(--magenta) 60%, var(--violet))
  );
  box-shadow:
    0 2px 8px color-mix(in oklab, var(--violet) 35%, transparent),
    0 0 0 1px color-mix(in oklab, var(--violet) 20%, transparent);
  transition: transform 320ms cubic-bezier(0.34, 1.56, 0.64, 1);

  // Plain `[data-theme] &`, NOT `:global([data-theme]) &`: the :global() form
  // drops the trailing `&` in scoped-CSS compilation, leaking `transform` onto
  // <html>, which then becomes the containing block for every position:fixed
  // descendant — so the fixed presence-bar scrolls away with the page.
  [data-theme='light'] & {
    transform: translateX(26px);
  }
}
</style>
