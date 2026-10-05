<script setup lang="ts">
/**
 * Slide-out side drawer shell — the scrim, right-anchored panel, header
 * (title + close), and scrollable list region shared by every right-side
 * overlay (the Activity drawer, the Scheduler dock). Consumers pass a title
 * and their list rows via the default slot; this component owns the open/close
 * transition choreography and Escape-to-close, so the rows are all a consumer
 * has to think about.
 *
 * Controlled: `open` drives the slide; a close request (scrim, close button, or
 * Escape) is reported via `close` — the parent flips its own state, which flows
 * back down through `open`. The enter path un-hides then adds `.open` on the next
 * frame so the transform animates; the exit path drops `.open` and re-hides only
 * on `transitionend`, so the panel leaves the layout after the slide completes,
 * not before.
 */
import { onBeforeUnmount, onMounted, ref, watch } from 'vue';
import { X } from '@lucide/vue';

const props = defineProps<{
  open: boolean;
  title: string;
}>();

const emit = defineEmits<{
  (e: 'close'): void;
}>();

const drawerRef = ref<HTMLElement | null>(null);
const scrimRef = ref<HTMLElement | null>(null);

function openDrawerDom(): void {
  if (!scrimRef.value || !drawerRef.value) return;
  scrimRef.value.classList.remove('hidden');
  drawerRef.value.classList.remove('hidden');
  requestAnimationFrame(() => {
    drawerRef.value?.classList.add('open');
  });
}

function closeDrawerDom(): void {
  const drawer = drawerRef.value;
  const scrim = scrimRef.value;
  if (!drawer) return;

  drawer.classList.remove('open');
  scrim?.classList.add('hidden');

  drawer.addEventListener(
    'transitionend',
    () => {
      if (!drawer.classList.contains('open')) {
        drawer.classList.add('hidden');
      }
    },
    { once: true },
  );
}

watch(
  () => props.open,
  (open) => {
    if (open) openDrawerDom();
    else closeDrawerDom();
  },
);

function handleKeydown(e: KeyboardEvent): void {
  if (e.key === 'Escape' && props.open) emit('close');
}

onMounted(() => {
  document.addEventListener('keydown', handleKeydown);
});

onBeforeUnmount(() => {
  document.removeEventListener('keydown', handleKeydown);
});
</script>

<template>
  <!-- Scrim -->
  <div
    ref="scrimRef"
    class="side-drawer__scrim hidden"
    aria-hidden="true"
    @click="emit('close')"
  ></div>

  <!-- Slide-out panel -->
  <aside ref="drawerRef" class="side-drawer hidden" :aria-label="title">
    <div class="side-drawer__header">
      <h2 class="side-drawer__title">{{ title }}</h2>
      <button
        class="btn-icon side-drawer__close"
        :aria-label="`Close ${title.toLowerCase()} panel`"
        @click="emit('close')"
      >
        <X :size="16" aria-hidden="true" />
      </button>
    </div>

    <div class="side-drawer__list">
      <slot />
    </div>
  </aside>
</template>

<style scoped lang="scss">
// ── Scrim ──────────────────────────────────────────────────────────────────────
// A transparent click-catcher: a click anywhere outside the panel closes it.

.side-drawer__scrim {
  position: fixed;
  inset: 0;
  z-index: 199;

  &.hidden {
    display: none;
  }
}

// ── Drawer panel ───────────────────────────────────────────────────────────────

.side-drawer {
  position: fixed;
  top: 0;
  right: 0;
  height: 100%;
  width: 320px;
  max-width: 90vw;
  background: var(--surface);
  border-left: 1px solid var(--line);
  z-index: 200;
  display: flex;
  flex-direction: column;
  transform: translateX(100%);
  transition: transform var(--dur-3) var(--ease-out);
  overflow: hidden;

  &.open {
    transform: translateX(0);
  }

  &.hidden {
    display: none;
  }
}

.side-drawer__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 16px 12px;
  border-bottom: 1px solid var(--line);
  flex-shrink: 0;
}

.side-drawer__title {
  margin: 0;
  font-size: 15px;
  font-weight: 600;
  color: var(--text);
}

.side-drawer__close {
  color: var(--muted);

  &:hover {
    color: var(--text);
  }
}

// ── List ───────────────────────────────────────────────────────────────────────

.side-drawer__list {
  flex: 1;
  overflow-y: auto;
  padding: 8px 0;
}
</style>
