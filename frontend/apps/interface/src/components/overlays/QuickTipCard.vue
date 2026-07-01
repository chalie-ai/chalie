<script setup lang="ts">
/**
 * QuickTipCard — slide-up feature-discovery card above the input dock.
 *
 * DORMANT: the backend does not currently emit `quick_tip` WS events; ported for
 * parity, activates automatically when the backend ships the event.
 *
 * State is owned by useNotificationsStore; this component is purely presentational.
 */
import { computed, onBeforeUnmount, onMounted } from 'vue';
import { Lightbulb, X } from '@lucide/vue';
import { useNotificationsStore } from '../../stores/notifications';

const notifications = useNotificationsStore();

const tip = computed(() => notifications.currentTip);
const visible = computed(() => tip.value !== null);

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
</script>

<template>
  <Transition name="tip">
    <div v-if="visible" class="tip tip--visible" role="status" aria-live="polite">
      <button class="tip__dismiss" aria-label="Dismiss tip" @click="notifications.dismissTip()">
        <X :size="12" :stroke-width="2.4" aria-hidden="true" />
      </button>

      <div class="tip__head">
        <!-- eslint-disable-next-line vue/no-v-html -->
        <div v-if="tip?.icon_svg" class="tip__icon" v-html="tip.icon_svg" />
        <div v-else class="tip__icon">
          <Lightbulb aria-hidden="true" />
        </div>
        <div class="tip__label">Quick tip</div>
      </div>

      <p class="tip__body">{{ tip?.body ?? '' }}</p>

      <div v-if="tip?.example" class="tip__example">
        <div class="tip__example-label">Try saying</div>
        <div class="tip__example-prompt">{{ tip.example }}</div>
      </div>

      <div class="tip__foot">
        <button class="tip__mute" @click="notifications.muteTip()">Don't show more tips</button>
      </div>
    </div>
  </Transition>
</template>

<style scoped lang="scss">
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
  // Shadow rgba(0,0,0,…)/rgba(255,255,255,…) are theme-neutral transparency — kept hardcoded, not tokenized, for identical shadow parity in both themes.
  box-shadow:
    0 1px 0 rgba(255, 255, 255, 0.04) inset,
    0 2px 8px rgba(0, 0, 0, 0.2),
    0 8px 32px rgba(0, 0, 0, 0.35);

  // Plain `[data-theme] &` — :global() drops the `&` and leaks these onto <html>.
  [data-theme='light'] & {
    background: color-mix(in oklab, var(--bg-2, #f5f3f0) 95%, transparent);
    box-shadow:
      0 1px 0 rgba(0, 0, 0, 0.04) inset,
      0 2px 8px rgba(0, 0, 0, 0.08),
      0 8px 32px rgba(0, 0, 0, 0.12);
  }
}

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
  color: var(--violet-bright);
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

.tip__body {
  font-size: 0.92rem;
  color: var(--text-primary, #eae6f2);
  line-height: 1.5;
  margin: 0 0 12px;
  letter-spacing: -0.005em;
}

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
  color: var(--violet-bright);
  font-style: italic;
  letter-spacing: -0.005em;

  &::before {
    content: '\201C';
  }
  &::after {
    content: '\201D';
  }
}

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

// Static resting position/opacity for the always-present "tip--visible" class,
// so the card sits correctly once the Transition's enter phase completes.
.tip--visible {
  opacity: 1;
  transform: translateX(-50%) translateY(0);
  pointer-events: auto;
}

.tip-enter-active,
.tip-leave-active {
  transition:
    opacity 300ms cubic-bezier(0.16, 1, 0.3, 1),
    transform 300ms cubic-bezier(0.16, 1, 0.3, 1);
}

.tip-enter-from {
  opacity: 0;
  transform: translateX(-50%) translateY(20px);
}

.tip-enter-to {
  opacity: 1;
  transform: translateX(-50%) translateY(0);
}

.tip-leave-to {
  opacity: 0;
  transform: translateX(-50%) translateY(-12px);
}
</style>
