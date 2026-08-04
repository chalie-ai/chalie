<script setup lang="ts">
/**
 * ImagePreviewModal — full-screen lightbox for an attached image.
 *
 * Teleported to <body> so it overlays the whole viewport regardless of where the
 * triggering bubble sits. Closes on backdrop click, the × button, or Escape.
 */
import { computed, onBeforeUnmount, onMounted, ref } from 'vue';

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
        ×
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
// The dim is intentionally dark in both themes — the image is the focus, the way
// every lightbox dims its surround regardless of the page theme.
.img-modal {
  position: fixed;
  inset: 0;
  z-index: 1200;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: var(--space-xl);
  background: rgba(0, 0, 0, 0.82);
  backdrop-filter: blur(8px);
  -webkit-backdrop-filter: blur(8px);
  animation: imgModalFade 160ms ease;
}

.img-modal__img {
  object-fit: contain;
  border-radius: var(--radius-md);
  box-shadow: 0 24px 64px rgba(0, 0, 0, 0.6);
}

.img-modal__close {
  position: absolute;
  top: var(--space-md);
  right: var(--space-md);
  display: flex;
  align-items: center;
  justify-content: center;
  width: 40px;
  height: 40px;
  border: none;
  border-radius: 50%;
  background: rgba(0, 0, 0, 0.45);
  color: #fff;
  font-size: 24px;
  line-height: 1;
  cursor: pointer;
  transition: background 150ms ease;

  &:hover {
    background: rgba(0, 0, 0, 0.7);
  }
}

@keyframes imgModalFade {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}
</style>
