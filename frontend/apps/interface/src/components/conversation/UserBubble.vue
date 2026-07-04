<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue';
import { FileText, Image as ImageIcon } from '@lucide/vue';
import type { ConversationAttachment, ConversationMessage } from '../../api/conversation';
import ImagePreviewModal from '../overlays/ImagePreviewModal.vue';

const props = defineProps<{ message: ConversationMessage }>();

// File-only messages carry the '[File attached]' placeholder text — drop the
// empty bubble and let the attachment list stand on its own.
const showText = computed(
  () => !(props.message.content === '[File attached]' && props.message.attachments?.length),
);

// Long messages collapse to a clamped 5-line preview with a show more/less toggle.
const textEl = ref<HTMLElement | null>(null);
const overflowing = ref(false);
const collapsed = ref(true);

function measure(): void {
  const el = textEl.value;
  if (!el) return;
  const lineHeight = Number.parseFloat(getComputedStyle(el).lineHeight) || 24;
  overflowing.value = el.scrollHeight > lineHeight * 5 + 1;
}

let resizeObserver: ResizeObserver | null = null;
onMounted(() => {
  void nextTick(measure);
  resizeObserver = new ResizeObserver(measure);
  if (textEl.value) resizeObserver.observe(textEl.value);
});
onBeforeUnmount(() => resizeObserver?.disconnect());
watch(
  () => props.message.content,
  () => {
    collapsed.value = true;
    void nextTick(measure);
  },
);

const previewSrc = ref<string | null>(null);
const previewAlt = ref('');

// Image → lightbox; document → download.
function open(att: ConversationAttachment): void {
  if (!att.url) return;
  if (att.is_image) {
    previewSrc.value = att.url;
    previewAlt.value = att.filename;
    return;
  }
  const a = document.createElement('a');
  a.href = att.url;
  a.download = att.filename || 'attachment';
  a.rel = 'noopener';
  document.body.appendChild(a);
  a.click();
  a.remove();
}
</script>

<template>
  <div class="user-message">
    <div
      v-if="showText"
      ref="textEl"
      class="user-text"
      :class="{ 'user-text--clamped': overflowing && collapsed }"
    >
      {{ message.content }}
    </div>

    <button
      v-if="showText && overflowing"
      type="button"
      class="user-text__toggle"
      @click="collapsed = !collapsed"
    >
      {{ collapsed ? 'show more' : 'show less' }}
    </button>

    <!-- Attachments rendered from the API model directly. -->
    <ul v-if="message.attachments && message.attachments.length" class="user-attachments">
      <li v-for="att in message.attachments" :key="att.filename">
        <button type="button" class="user-attachments__item" @click="open(att)">
          <component
            :is="att.is_image ? ImageIcon : FileText"
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
