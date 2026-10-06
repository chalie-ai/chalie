<script setup lang="ts">
/**
 * ImagePreviewModal — full-screen lightbox for an attached image.
 *
 * Teleported to <body> so it overlays the whole viewport regardless of where the
 * triggering bubble sits. Closes on backdrop click, the × button, or Escape.
 */
import { computed, onBeforeUnmount, onMounted, ref } from 'vue';
import { X } from '@lucide/vue';

interface Props {
  src: string;
  alt?: string;
  maxW?: string;
  maxH?: string;
  /**
   * Scale the image UP as well as down, until one side meets its cap. Off by
   * default: an attachment opens at its own size, the way it was sent.
   */
  fill?: boolean;
}

const props = withDefaults(defineProps<Props>(), {
  alt: () => 'attached',
  maxW: () => '100%',
  maxH: () => '100%',
  fill: false,
});

const emit = defineEmits<{ close: [] }>();

// 0 until the bitmap reports its size; an SVG with no intrinsic size never does,
// and falls back to the plain caps below.
const ratio = ref(0);

// Fill mode makes the longer side the binding one — width = min(width cap,
// height cap × ratio) — so the image grows to a cap without ever being cropped
// or distorted. Plain mode just caps both sides.
const fitStyle = computed(() =>
  props.fill && ratio.value
    ? { width: `min(${props.maxW}, calc(${props.maxH} * ${ratio.value}))` }
    : { maxWidth: props.maxW, maxHeight: props.maxH },
);

function onLoad(e: Event): void {
  const img = e.target as HTMLImageElement;
  if (img.naturalWidth && img.naturalHeight) ratio.value = img.naturalWidth / img.naturalHeight;
}

function onKey(e: KeyboardEvent): void {
  if (e.key === 'Escape') emit('close');
}
onMounted(() => document.addEventListener('keydown', onKey));
onBeforeUnmount(() => document.removeEventListener('keydown', onKey));
</script>

<template>
  <Teleport to="body">
    <div class="img-modal" role="dialog" aria-modal="true" @click.self="emit('close')">
      <button
        class="img-modal__close"
        type="button"
        aria-label="Close preview"
        @click="emit('close')"
      >
        <X :size="16" />
      </button>
      <img
        class="img-modal__img"
        :src="src"
        :alt="alt"
        :style="fitStyle"
        referrerpolicy="no-referrer"
        @load="onLoad"
      />
    </div>
  </Teleport>
</template>

<style scoped lang="scss">
.img-modal {
  position: fixed;
  inset: 0;
  z-index: 1200;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: var(--space-xl);
  background: var(--bg);
  animation: fade var(--dur-2) var(--ease-out);
}

.img-modal__img {
  object-fit: contain;
}

.img-modal__close {
  position: absolute;
  top: var(--space-md);
  right: var(--space-md);
  display: flex;
  align-items: center;
  justify-content: center;
  width: var(--control-h);
  height: var(--control-h);
  border: none;
  background: var(--surface-2);
  color: var(--text);
  cursor: pointer;
  transition:
    background-color var(--dur-1) var(--ease-out),
    translate var(--dur-1) var(--ease-out);

  &:hover {
    background: var(--cell);
  }

  &:active {
    translate: 0 1px;
  }
}
</style>
