<script setup lang="ts">
import { Star, Clock, SlidersHorizontal, Sun, Moon } from '@lucide/vue';
import { storeToRefs } from 'pinia';
import { usePresenceStore } from '../../stores/presence';
import { useTasksStore } from '../../stores/tasks';
import { useTheme } from '@chalie/shared';
import { emit } from '../../composables/useEventBus';

const presence = usePresenceStore();
const { state, label } = storeToRefs(presence);

const tasks = useTasksStore();
const { totalCount } = storeToRefs(tasks);

const { toggle, theme } = useTheme();

function handleThemeToggle(): void {
  toggle();
  emit('chalie:theme-changed', { theme: theme.value });
}

/** Settings button → open the Brain admin dashboard (port of app.js:266-267). */
function handleSettings(): void {
  globalThis.open('/brain/', 'chalie-brain');
}
</script>

<template>
  <header class="presence-bar">
    <div class="presence-bar__left">
      <div class="presence-dot" :data-state="state">
        <div class="presence-dot__inner"></div>
      </div>
      <span class="presence-label">{{ label }}</span>
    </div>
    <div class="presence-bar__right">
      <!-- Recall — opens the moment-search dialog (App.vue owns the dialog ref). -->
      <button
        id="recallBtn"
        class="btn-icon"
        aria-label="Recall"
        title="Recall"
        @click="emit('chalie:open-recall', {})"
      >
        <Star :size="18" />
      </button>
      <!-- Task drawer trigger — shown only when there is something to display.
           Port of legacy index.html lines 122-127 (taskDrawerBtn).
           Visibility matches task_strip.js _render(): trigger hidden when totalCount === 0. -->
      <button
        v-if="totalCount > 0"
        id="taskDrawerBtn"
        class="btn-icon task-drawer-trigger"
        aria-label="Active tasks"
        title="Active tasks"
        @click="tasks.open()"
      >
        <Clock :size="18" aria-hidden="true" />
        <span class="task-trigger__badge">{{ totalCount }}</span>
      </button>
      <!-- Settings — opens the Brain admin dashboard (app.js:266-267). -->
      <button
        id="settingsBtn"
        class="btn-icon"
        aria-label="Settings"
        @click="handleSettings"
      >
        <SlidersHorizontal :size="18" />
      </button>
      <button
        class="theme-toggle"
        :aria-label="'Toggle light or dark theme'"
        title="Toggle light / dark"
        @click="handleThemeToggle"
      >
        <span class="theme-toggle__track" aria-hidden="true">
          <!-- Sun icon -->
          <Sun />
          <!-- Moon icon -->
          <Moon />
        </span>
        <span class="theme-toggle__thumb" aria-hidden="true"></span>
      </button>
    </div>
  </header>
</template>

<style scoped lang="scss">
// --------------------------------------------------------------------------
// Task drawer trigger (presence-bar placement)
// --------------------------------------------------------------------------

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
  font-weight: 700;
  line-height: 16px;
  text-align: center;
  pointer-events: none;
}

// --------------------------------------------------------------------------
// Presence Dot + per-state animations
// Port of legacy style.css §6 (lines 479–602).
// All colors via CSS token vars so both dark and light themes work (Rule 7).
// --------------------------------------------------------------------------

.presence-dot {
  position: relative;
  width: 12px;
  height: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.presence-dot__inner {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: var(--sand);
  transition: all var(--duration-normal) var(--ease-out);
}

.presence-label {
  font-size: var(--font-size-sm);
  font-weight: 500;
  color: var(--text-secondary);
  transition: color var(--duration-normal) var(--ease-out);
}

// ── Resting — breathe ──────────────────────────────────────────────────────
.presence-dot[data-state="resting"] .presence-dot__inner {
  animation: pb-breathe 4s ease-in-out infinite;
  background: var(--sand);
  box-shadow: 0 0 8px currentColor;
}

@keyframes pb-breathe {
  0%, 100% { transform: scale(1);   opacity: 0.7; }
  50%       { transform: scale(1.2); opacity: 1;   }
}

// ── Processing — pulse ─────────────────────────────────────────────────────
.presence-dot[data-state="processing"] .presence-dot__inner {
  animation: pb-pulse 1.5s ease-in-out infinite;
  background: var(--dusk-blue);
  box-shadow: 0 0 8px currentColor;
}

@keyframes pb-pulse {
  0%, 100% { transform: scale(1);   }
  50%       { transform: scale(1.4); }
}

// ── Thinking — glow ────────────────────────────────────────────────────────
.presence-dot[data-state="thinking"] .presence-dot__inner {
  animation: pb-glow 2s ease-in-out infinite;
  background: var(--violet);
}

@keyframes pb-glow {
  0%, 100% { box-shadow: 0 0 4px var(--violet); }
  50%       { box-shadow: 0 0 16px var(--violet), 0 0 32px color-mix(in oklab, var(--violet) 30%, transparent); }
}

// ── Retrieving memory — ripple ─────────────────────────────────────────────
.presence-dot[data-state="retrieving_memory"] .presence-dot__inner {
  animation: pb-ripple 2s ease-out infinite;
  background: var(--dusk-blue);
  box-shadow: 0 0 8px currentColor;
}

@keyframes pb-ripple {
  0%   { box-shadow: 0 0 0 0   color-mix(in oklab, var(--cyan) 40%, transparent); }
  70%  { box-shadow: 0 0 0 12px color-mix(in oklab, var(--cyan)  0%, transparent); }
  100% { box-shadow: 0 0 0 0   color-mix(in oklab, var(--cyan)  0%, transparent); }
}

// ── Planning — shimmer ─────────────────────────────────────────────────────
.presence-dot[data-state="planning"] .presence-dot__inner {
  background: linear-gradient(90deg, var(--violet), var(--dusk-blue), var(--violet));
  background-size: 200% 100%;
  animation: pb-shimmer 2s ease-in-out infinite;
}

@keyframes pb-shimmer {
  0%   { background-position:  200% 0; }
  100% { background-position: -200% 0; }
}

// ── Narrating — shimmer (violet↔cyan, 3s) ─────────────────────────────────
.presence-dot[data-state="narrating"] .presence-dot__inner {
  background: linear-gradient(90deg, var(--violet), var(--dusk-blue), var(--violet));
  background-size: 200% 100%;
  animation: pb-shimmer 3s ease-in-out infinite;
}

// ── Responding — waveform (dot widens to bar) ──────────────────────────────
.presence-dot[data-state="responding"] {
  width: 24px;
}

.presence-dot[data-state="responding"] .presence-dot__inner {
  width: 24px;
  height: 6px;
  border-radius: 3px;
  background: var(--amber);
  animation: pb-waveform 0.8s ease-in-out infinite alternate;
}

@keyframes pb-waveform {
  0%   { transform: scaleY(1);   }
  25%  { transform: scaleY(1.8); }
  50%  { transform: scaleY(0.6); }
  75%  { transform: scaleY(1.4); }
  100% { transform: scaleY(1);   }
}

// ── Still working — same as processing ────────────────────────────────────
.presence-dot[data-state="still_working"] .presence-dot__inner {
  animation: pb-pulse 1.5s ease-in-out infinite;
  background: var(--dusk-blue);
  box-shadow: 0 0 8px currentColor;
}

// ── Error — blink ──────────────────────────────────────────────────────────
.presence-dot[data-state="error"] .presence-dot__inner {
  background: var(--error);
  animation: pb-blink 1.5s infinite;
}

@keyframes pb-blink {
  0%, 100% { opacity: 1;   }
  50%       { opacity: 0.3; }
}

// --------------------------------------------------------------------------
// Theme toggle — sliding pill with sun/moon glyphs
// Port of legacy style.css §5 theme-toggle block (lines 429–474).
// --------------------------------------------------------------------------
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
  transition: background 220ms ease, border-color 220ms ease;

  &:hover { border-color: var(--border-strong); }

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
  background: linear-gradient(135deg, var(--violet), color-mix(in oklab, var(--magenta) 60%, var(--violet)));
  box-shadow:
    0 2px 8px color-mix(in oklab, var(--violet) 35%, transparent),
    0 0 0 1px color-mix(in oklab, var(--violet) 20%, transparent);
  transition: transform 320ms cubic-bezier(0.34, 1.56, 0.64, 1);

  // NB: plain `[data-theme] &` — NOT `:global([data-theme]) &`. The :global()
  // form drops the trailing `&` during scoped-CSS compilation, leaking
  // `transform` onto the bare `[data-theme="light"]` (the <html> element). A
  // transform on <html> makes it the containing block for every position:fixed
  // descendant, so the fixed presence-bar scrolls away with the page.
  [data-theme="light"] & {
    transform: translateX(26px);
  }
}
</style>
