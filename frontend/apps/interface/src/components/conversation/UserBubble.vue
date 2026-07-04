<script setup lang="ts">
import { computed, ref } from 'vue';
import { FileText, Image as ImageIcon } from '@lucide/vue';
import type { AttachmentPreview, UserForm } from '../../stores/conversation';
import ImagePreviewModal from '../overlays/ImagePreviewModal.vue';

const props = defineProps<{ form: UserForm }>();

// File-only messages carry the '[File attached]' placeholder text — drop the
// empty bubble and let the attachment list stand on its own.
const showText = computed(
  () => !(props.form.text === '[File attached]' && props.form.attachments?.length),
);

const previewSrc = ref<string | null>(null);
const previewAlt = ref('');

// Image → lightbox; document → download. objectUrl is a same-origin URL (history)
// or a data URL (just-sent), both of which honour the anchor download attribute.
function open(att: AttachmentPreview): void {
  if (!att.objectUrl) return;
  if (att.isImage) {
    previewSrc.value = att.objectUrl;
    previewAlt.value = att.filename;
    return;
  }
  const a = document.createElement('a');
  a.href = att.objectUrl;
  a.download = att.filename || 'attachment';
  a.rel = 'noopener';
  document.body.appendChild(a);
  a.click();
  a.remove();
}
</script>

<template>
  <div
    class="user-message"
    :class="{ 'message--faded': form.inWorkingMemory === false }"
  >
    <div v-if="showText" class="speech-form speech-form--user">{{ form.text }}</div>

    <div v-if="form.ts" class="user-message__ts">{{ form.ts }}</div>

    <!-- Attachments live OUTSIDE the bubble, below it: a condensed, right-aligned
         list in the act-trail idiom. Click an image to preview, a doc to download. -->
    <ul v-if="form.attachments && form.attachments.length" class="user-attachments">
      <li v-for="att in form.attachments" :key="att.filename">
        <button type="button" class="user-attachments__item" @click="open(att)">
          <component
            :is="att.isImage ? ImageIcon : FileText"
            class="user-attachments__icon"
            :size="13"
          />
          <span class="user-attachments__name">{{ att.filename || 'attachment' }}</span>
        </button>
      </li>
    </ul>

    <ImagePreviewModal
      v-if="previewSrc"
      :src="previewSrc"
      :alt="previewAlt"
      @close="previewSrc = null"
    />
  </div>
</template>
