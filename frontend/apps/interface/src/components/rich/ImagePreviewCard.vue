<script setup lang="ts">
/**
 * ImagePreviewCard — rendered alongside chat messages to show a single high-
 * quality preview image chosen by the assistant, with an optional caption.
 * Clicking the thumbnail opens ImagePreviewModal in a larger lightbox.
 */
import { computed, ref } from 'vue';
import ImagePreviewModal from '../overlays/ImagePreviewModal.vue';

export interface ImagePreviewPayload {
  url: string;
  alt?: string;
  subtitle?: string;
}

// In-line thumbnails scale to 640px on their longer side (or 90vw, whichever
// lands first). The lightbox caps live on the modal's call site below.
const THUMB_CAP = 640;

defineProps<{
  payload: ImagePreviewPayload;
  // synthesis satisfies the shared rich-card prop contract; not rendered here
  // because the payload's own subtitle owns this card's caption slot.
  synthesis?: string;
}>();

const isOpen = ref(false);

// Aspect ratio of the loaded bitmap, 0 until it is known. Sizing needs it: the
// image scales until ONE side meets its cap, so which side is the binding one
// depends on the shape. Capping both sides instead would letterbox a portrait
// image inside a 640-wide box — a visible gap with the caption adrift under it.
const ratio = ref(0);

// width = min(width cap, height cap × ratio) makes the taller side the binding
// one for a portrait image and the wider side for a landscape one, with height
// left to follow. Nothing is ever cropped or distorted.
const thumbStyle = computed(() =>
  ratio.value ? { width: `min(${THUMB_CAP}px, 90vw, ${THUMB_CAP * ratio.value}px)` } : undefined,
);

function onLoad(e: Event): void {
  const img = e.target as HTMLImageElement;
  // An SVG with no intrinsic size reports 0×0; leave the CSS default in charge.
  if (img.naturalWidth && img.naturalHeight) ratio.value = img.naturalWidth / img.naturalHeight;
}

function open(): void {
  isOpen.value = true;
}

function close(): void {
  isOpen.value = false;
}
</script>

<template>
  <div class="rich-card image-preview-card">
    <img
      class="image-preview-card__img"
      :src="payload.url"
      :alt="payload.alt || 'preview'"
      :style="thumbStyle"
      @load="onLoad"
      @click="open"
    />
    <p v-if="payload.subtitle" class="image-preview-card__caption">
      {{ payload.subtitle }}
    </p>
    <ImagePreviewModal
      v-if="isOpen"
      :src="payload.url"
      :alt="payload.alt || 'preview'"
      max-w="90vw"
      max-h="70vh"
      fill
      @close="close"
    />
  </div>
</template>

<style scoped lang="scss">
.image-preview-card__img {
  width: min(640px, 90vw);
  height: auto;
  max-height: 640px;
  object-fit: contain;
  cursor: pointer;
  border-radius: var(--radius-md);
  display: block;
}

.image-preview-card__caption {
  margin: 10px 0 0;
  padding: 0 16px;
  color: var(--text-secondary);
  font-size: 0.8125rem;
  line-height: 1.4;
}
</style>
