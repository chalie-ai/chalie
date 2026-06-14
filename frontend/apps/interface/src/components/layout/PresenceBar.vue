<script setup lang="ts">
import { storeToRefs } from 'pinia';
import { usePresenceStore } from '../../stores/presence';
import { useTheme } from '@chalie/shared';
import { emit } from '../../composables/useEventBus';

const presence = usePresenceStore();
const { state, label } = storeToRefs(presence);

const { toggle, theme } = useTheme();

function handleThemeToggle(): void {
  toggle();
  emit('chalie:theme-changed', { theme: theme.value });
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
      <button
        class="theme-toggle"
        :aria-label="'Toggle light or dark theme'"
        title="Toggle light / dark"
        @click="handleThemeToggle"
      >
        <span class="theme-toggle__track" aria-hidden="true">
          <!-- Sun icon -->
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="4"/>
            <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"/>
          </svg>
          <!-- Moon icon -->
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
          </svg>
        </span>
        <span class="theme-toggle__thumb" aria-hidden="true"></span>
      </button>
    </div>
  </header>
</template>

<style scoped lang="scss">
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

  :global([data-theme="light"]) & {
    transform: translateX(26px);
  }
}
</style>
