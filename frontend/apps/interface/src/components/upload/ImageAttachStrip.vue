<script setup lang="ts">
/**
 * ImageAttachStrip — reactive preview strip for image and document attachments.
 *
 * Reads the attachments store; does NOT own any <input type=file> (those are in
 * InputDock). Drag-drop and paste are wired on document so the entire viewport
 * acts as a drop target. Strip is hidden when there are no previews.
 */
import { computed, onBeforeUnmount, onMounted, ref } from 'vue';
import { useAttachmentsStore } from '../../stores/attachments';

const attachments = useAttachmentsStore();

const hasPreviews = computed(() => attachments.previews.length > 0);

function truncate(name: string): string {
  return name.length > 20 ? name.slice(0, 18) + '…' : name;
}

const dragging = ref(false);

function onDragEnter(ev: DragEvent): void {
  if (!ev.dataTransfer?.types?.includes('Files')) return;
  ev.preventDefault();
  dragging.value = true;
}

function onDragOver(ev: DragEvent): void {
  if (!ev.dataTransfer?.types?.includes('Files')) return;
  ev.preventDefault();
}

function onDragLeave(ev: DragEvent): void {
  // relatedTarget is null only when leaving to outside the document.
  if (ev.relatedTarget) return;
  dragging.value = false;
}

function onDrop(ev: DragEvent): void {
  ev.preventDefault();
  dragging.value = false;
  const files = ev.dataTransfer?.files;
  if (!files?.length) return;
  void attachments.addFiles(files);
}

function onPaste(ev: ClipboardEvent): void {
  const items = ev.clipboardData?.items;
  if (!items) return;
  for (let i = 0; i < items.length; i++) {
    const item = items[i];
    if (item.type.startsWith('image/')) {
      const file = item.getAsFile();
      if (!file) continue;
      ev.preventDefault();
      void attachments.addFiles([file]);
      return;
    }
  }
}

onMounted(() => {
  document.addEventListener('dragenter', onDragEnter);
  document.addEventListener('dragover', onDragOver);
  document.addEventListener('dragleave', onDragLeave);
  document.addEventListener('drop', onDrop);
  document.addEventListener('paste', onPaste);
});

onBeforeUnmount(() => {
  document.removeEventListener('dragenter', onDragEnter);
  document.removeEventListener('dragover', onDragOver);
  document.removeEventListener('dragleave', onDragLeave);
  document.removeEventListener('drop', onDrop);
  document.removeEventListener('paste', onPaste);
});
</script>

<template>
  <div
    v-show="hasPreviews"
    id="imagePreview"
    class="image-preview"
    :class="{ 'image-preview--drag': dragging }"
  >
    <div
      v-for="(preview, index) in attachments.previews"
      :key="index"
      class="image-preview__thumb"
      :class="{ 'image-preview__thumb--doc': !preview.isImage }"
    >
      <template v-if="preview.isImage">
        <img
          v-if="preview.dataUrl"
          :src="preview.dataUrl"
          :alt="preview.filename"
        />
      </template>

      <template v-else>
        <div class="image-preview__doc-icon" aria-hidden="true">📄</div>
        <span class="image-preview__doc-name">{{ truncate(preview.filename) }}</span>
      </template>

      <button
        class="image-preview__remove"
        type="button"
        aria-label="Remove attachment"
        @click="attachments.remove(index)"
      >
        ×
      </button>
    </div>
  </div>
</template>

<style scoped lang="scss">
.image-preview {
  // Docked tray: a panel centred over the input box, sharing its width and
  // sitting just above it — not a strip stranded at the dock's left edge.
  max-width: var(--dock-width);
  margin: 0 auto var(--space-sm);
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-xs, 6px);
  padding: var(--space-sm);
  background: var(--bg-surface-2);
  border: 1px solid var(--border-strong);
  border-radius: var(--radius-md);
  backdrop-filter: blur(20px) saturate(120%);
  -webkit-backdrop-filter: blur(20px) saturate(120%);

  &--drag {
    outline: 2px dashed var(--accent, currentColor);
    outline-offset: -2px;
  }

  &__thumb {
    position: relative;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 56px;
    height: 56px;
    border-radius: var(--radius-sm, 6px);
    overflow: hidden;
    background: var(--surface-raised, var(--surface, #222));
    border: 1px solid var(--border, rgba(255 255 255 / 0.12));
    flex-shrink: 0;

    img {
      width: 100%;
      height: 100%;
      object-fit: cover;
    }

    &--doc {
      flex-direction: column;
      gap: 2px;
      padding: 4px;
      width: auto;
      min-width: 64px;
      max-width: 120px;
      // A doc thumb shows its frame (unlike image thumbs), so it needs a real
      // theme-aware token: --surface-raised/--surface don't exist here and fell
      // back to #222 — illegible against the light-mode filename text.
      background: var(--bg-surface-2);
    }
  }

  &__doc-icon {
    font-size: 18px;
    line-height: 1;
  }

  &__doc-name {
    font-size: 10px;
    color: var(--text-secondary, var(--text, currentColor));
    text-align: center;
    word-break: break-all;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 100%;
  }

  &__remove {
    position: absolute;
    top: 2px;
    right: 2px;
    display: flex;
    align-items: center;
    justify-content: center;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    border: none;
    padding: 0;
    font-size: 12px;
    line-height: 1;
    cursor: pointer;
    background: var(--surface-overlay, rgba(0 0 0 / 0.6));
    color: var(--text, currentColor);
    opacity: 0;
    transition: opacity 0.15s;

    .image-preview__thumb:hover & {
      opacity: 1;
    }
  }
}
</style>
