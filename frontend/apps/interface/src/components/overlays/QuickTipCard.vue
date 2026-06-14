<script setup lang="ts">
/**
 * QuickTipCard — slide-up feature-discovery card above the input dock.
 *
 * Port of frontend/interface/quick_tip_card.js.
 *
 * DORMANT: the backend does not currently emit `quick_tip` WS events. This
 * component is ported for parity and will activate automatically when the
 * backend ships the event.
 *
 * State is owned by useNotificationsStore. The component is purely presentational:
 *   Dismiss (✕ / Escape) → notifications.dismissTip()
 *   Mute                 → notifications.muteTip()
 */
import { computed, onMounted, onBeforeUnmount } from 'vue';
import { useNotificationsStore } from '../../stores/notifications';

const notifications = useNotificationsStore();

const tip = computed(() => notifications.currentTip);
const visible = computed(() => tip.value !== null);

// ── Escape key dismiss (port of quick_tip_card.js lines 88-90) ───────────────

function onKeydown(e: KeyboardEvent): void {
  if (e.key === 'Escape' && visible.value) {
    notifications.dismissTip();
  }
}

onMounted(() => {
  document.addEventListener('keydown', onKeydown);
});

onBeforeUnmount(() => {
  document.removeEventListener('keydown', onKeydown);
});

// ── Fallback icon SVG (port of quick_tip_card.js _fallbackIcon(), lines 162-168) ──

const fallbackIconSvg = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
  stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
  <path d="M9 18h6"/><path d="M10 22h4"/>
  <path d="M12 2a7 7 0 0 0-4 12.7V17h8v-2.3A7 7 0 0 0 12 2z"/>
</svg>`;
</script>

<template>
  <!--
    Port of the .tip element built in quick_tip_card.js _buildCard().
    Visibility is driven by the `tip--visible` / `tip--leaving` class pattern
    from the legacy CSS, replicated here via Transition + CSS.
    aria-hidden mirrors the legacy setAttribute('aria-hidden', ...) pattern.
  -->
  <Transition name="tip">
    <div
      v-if="visible"
      class="tip tip--visible"
      role="status"
      aria-live="polite"
    >
      <!-- Dismiss ✕ (port of .tip__dismiss button, lines 56-62) -->
      <button
        class="tip__dismiss"
        aria-label="Dismiss tip"
        @click="notifications.dismissTip()"
      >
        <svg
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2.4"
          stroke-linecap="round"
          aria-hidden="true"
        >
          <line x1="18" y1="6" x2="6" y2="18" />
          <line x1="6" y1="6" x2="18" y2="18" />
        </svg>
      </button>

      <!-- Header: icon + label (port of .tip__head, lines 63-67) -->
      <div class="tip__head">
        <!-- eslint-disable-next-line vue/no-v-html -->
        <div class="tip__icon" v-html="tip?.icon_svg ?? fallbackIconSvg" />
        <div class="tip__label">Quick tip</div>
      </div>

      <!-- Body (port of .tip__body, line 68) -->
      <p class="tip__body">{{ tip?.body ?? '' }}</p>

      <!-- "Try saying" example (port of .tip__example, lines 69-72) -->
      <div v-if="tip?.example" class="tip__example">
        <div class="tip__example-label">Try saying</div>
        <div class="tip__example-prompt">{{ tip.example }}</div>
      </div>

      <!-- Footer: mute button (port of .tip__foot, lines 73-75) -->
      <div class="tip__foot">
        <button class="tip__mute" @click="notifications.muteTip()">
          Don't show more tips
        </button>
      </div>
    </div>
  </Transition>
</template>

<style scoped lang="scss">
/*
 * Styles are a scoped port of the .tip / .tip--visible / .tip--leaving rules
 * from frontend/interface/style.css (lines 3965-4131).
 * Theme-sensitive colors use CSS custom properties. Some box-shadow and inset
 * rgba values are theme-neutral (black transparency) and are ported verbatim
 * from the legacy source to preserve the exact visual output (Rule 7).
 */

.tip {
  position: fixed;
  left: 50%;
  bottom: calc(var(--input-dock-height, 80px) + var(--space-md, 16px));
  transform: translateX(-50%) translateY(20px);
  width: min(480px, calc(100vw - 32px));
  z-index: 190;
  padding: 18px 20px 16px;
  border-radius: var(--radius-md, 16px);
  background: color-mix(in oklab, var(--bg-2, #0d0f18) 95%, transparent);
  border: 1px solid var(--border-subtle, rgba(255, 255, 255, 0.07));
  border-top: 1px solid color-mix(in oklab, var(--accent-primary, #8a5cff) 55%, transparent);
  backdrop-filter: blur(16px) saturate(140%);
  -webkit-backdrop-filter: blur(16px) saturate(140%);
  box-shadow:
    0 1px 0 rgba(255, 255, 255, 0.04) inset,
    0 2px 8px rgba(0, 0, 0, 0.2),
    0 8px 32px rgba(0, 0, 0, 0.35);

  :global([data-theme='light']) & {
    background: color-mix(in oklab, var(--bg-2, #f5f3f0) 95%, transparent);
    box-shadow:
      0 1px 0 rgba(0, 0, 0, 0.04) inset,
      0 2px 8px rgba(0, 0, 0, 0.08),
      0 8px 32px rgba(0, 0, 0, 0.12);
  }
}

// ── Dismiss button (port of .tip__dismiss) ────────────────────────────────────

.tip__dismiss {
  position: absolute;
  top: 12px;
  right: 12px;
  width: 24px;
  height: 24px;
  border-radius: 6px;
  border: none;
  background: transparent;
  color: var(--text-tertiary, rgba(234, 230, 242, 0.3));
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  transition: all 160ms ease;

  &:hover {
    color: var(--text-primary, #eae6f2);
    background: var(--bg-surface, rgba(255, 255, 255, 0.03));
  }

  svg {
    width: 12px;
    height: 12px;
  }
}

// ── Header (port of .tip__head / .tip__icon / .tip__label) ───────────────────

.tip__head {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 10px;
}

.tip__icon {
  width: 28px;
  height: 28px;
  border-radius: 8px;
  background: color-mix(in oklab, var(--accent-primary, #8a5cff) 14%, transparent);
  border: 1px solid color-mix(in oklab, var(--accent-primary, #8a5cff) 28%, transparent);
  display: inline-flex;
  align-items: center;
  justify-content: center;
  color: #b07cff;
  flex-shrink: 0;

  :deep(svg) {
    width: 14px;
    height: 14px;
  }
}

.tip__label {
  font-family: var(--font-mono, 'JetBrains Mono', monospace);
  font-size: 0.62rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--text-tertiary, rgba(234, 230, 242, 0.3));
}

// ── Body (port of .tip__body) ─────────────────────────────────────────────────

.tip__body {
  font-size: 0.92rem;
  color: var(--text-primary, #eae6f2);
  line-height: 1.5;
  margin: 0 0 12px;
  letter-spacing: -0.005em;
}

// ── Example block (port of .tip__example / .tip__example-label / .tip__example-prompt) ──

.tip__example {
  padding: 10px 12px;
  border-radius: var(--radius-sm, 8px);
  background: color-mix(in oklab, var(--accent-primary, #8a5cff) 8%, transparent);
  border: 1px solid color-mix(in oklab, var(--accent-primary, #8a5cff) 20%, transparent);
  margin-bottom: 12px;
}

.tip__example-label {
  font-family: var(--font-mono, 'JetBrains Mono', monospace);
  font-size: 0.62rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--text-tertiary, rgba(234, 230, 242, 0.3));
  margin-bottom: 3px;
}

.tip__example-prompt {
  font-size: 0.88rem;
  color: #b07cff;
  font-style: italic;
  letter-spacing: -0.005em;

  &::before { content: '\201C'; }
  &::after  { content: '\201D'; }
}

// ── Footer / mute (port of .tip__foot / .tip__mute) ──────────────────────────

.tip__foot {
  display: flex;
  justify-content: flex-end;
  align-items: center;
  margin-top: 4px;
}

.tip__mute {
  background: transparent;
  border: none;
  padding: 4px 0;
  font-family: var(--font-sans, 'Inter', system-ui, sans-serif);
  font-size: 0.78rem;
  color: var(--text-tertiary, rgba(234, 230, 242, 0.3));
  cursor: pointer;
  transition: color 160ms ease;
  letter-spacing: -0.005em;

  &:hover {
    color: var(--text-secondary, rgba(234, 230, 242, 0.58));
  }
}

// ── Visible resting state (port of legacy style.css .tip--visible, lines 3989-3993) ──
// The element carries class="tip tip--visible" statically when visible.
// This rule establishes the resting position/opacity so the card sits correctly
// when the Transition has completed its enter phase.

.tip--visible {
  opacity: 1;
  transform: translateX(-50%) translateY(0);
  pointer-events: auto;
}

// ── Vue Transition — mirrors .tip--visible / .tip--leaving CSS animation ──────

.tip-enter-active,
.tip-leave-active {
  transition:
    opacity 300ms cubic-bezier(0.16, 1, 0.3, 1),
    transform 300ms cubic-bezier(0.16, 1, 0.3, 1);
}

// Enter: start translated down + invisible (port of base .tip: opacity:0, translateY(20px))
.tip-enter-from {
  opacity: 0;
  transform: translateX(-50%) translateY(20px);
}

// Enter-to: land at the resting state (translateY(0), full opacity).
.tip-enter-to {
  opacity: 1;
  transform: translateX(-50%) translateY(0);
}

// Leave: slide up + fade (port of .tip--leaving: translateY(-12px), opacity:0)
.tip-leave-to {
  opacity: 0;
  transform: translateX(-50%) translateY(-12px);
}
</style>
