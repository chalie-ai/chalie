<script setup lang="ts">
import { computed, ref, onMounted, onBeforeUnmount, nextTick, watch } from 'vue';
import { FileText, Image as ImageIcon } from '@lucide/vue';
import type { AttachmentPreview, UserForm } from '../../stores/conversation';
import ImagePreviewModal from '../overlays/ImagePreviewModal.vue';

const props = defineProps<{ form: UserForm }>();

// File-only messages carry the '[File attached]' placeholder text — drop the
// empty bubble and let the attachment list stand on its own.
const showText = computed(
  () => !(props.form.text === '[File attached]' && props.form.attachments?.length),
);

// Long messages collapse to a clamped 5-line preview with a show more/less
// toggle. Overflow is measured against the 5-line height (mirroring the CSS
// clamp) so it holds whether the text is currently clamped or expanded — and
// re-measures on width changes, since line count depends on the body width.
const textEl = ref<HTMLElement | null>(null);
const overflowing = ref(false);
const collapsed = ref(true);

function measure(): void {
  const el = textEl.value;
  if (!el) return;
  const lineHeight = parseFloat(getComputedStyle(el).lineHeight) || 24;
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
  () => props.form.text,
  () => {
    collapsed.value = true;
    void nextTick(measure);
  },
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
    <div
      v-if="showText"
      ref="textEl"
      class="user-text"
      :class="{ 'user-text--clamped': overflowing && collapsed }"
    >{{ form.text }}</div>

    <button
      v-if="showText && overflowing"
      type="button"
      class="user-text__toggle"
      @click="collapsed = !collapsed"
    >{{ collapsed ? 'show more' : 'show less' }}</button>

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
