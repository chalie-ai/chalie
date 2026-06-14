<script setup lang="ts">
import type { UserForm } from '../../stores/conversation';

defineProps<{ form: UserForm }>();
</script>

<template>
  <div
    class="speech-form speech-form--user"
    :class="{ 'message--faded': form.inWorkingMemory === false }"
  >
    <!-- Sender glyph — user spark, positioned at cap of accent stripe -->
    <svg class="sender-glyph" viewBox="0 0 18 18" fill="none" role="img" aria-label="You said">
      <title>You said</title>
      <path d="M9 2 L11 7 L16 9 L11 11 L9 16 L7 11 L2 9 L7 7 Z" fill="currentColor" opacity="0.85" />
    </svg>

    <!-- Attachment previews — ABOVE the text, matching legacy _appendUserAttachments -->
    <div
      v-if="form.attachments && form.attachments.length"
      class="speech-form__attachments"
    >
      <template v-for="att in form.attachments" :key="att.filename">
        <img
          v-if="att.isImage && att.objectUrl"
          class="speech-form__attachment-img"
          :src="att.objectUrl"
          :alt="att.filename || 'attached image'"
          loading="lazy"
        />
        <div v-else class="speech-form__attachment-doc">
          <span class="speech-form__attachment-doc-icon" aria-hidden="true">📄</span>
          <span class="speech-form__attachment-doc-name">{{ att.filename || 'attachment' }}</span>
        </div>
      </template>
    </div>

    <!-- User text — suppressed when '[File attached]' placeholder + attachments shown -->
    <div
      v-if="!(form.text === '[File attached]' && form.attachments && form.attachments.length)"
      class="speech-form__text"
    >{{ form.text }}</div>
  </div>
</template>
