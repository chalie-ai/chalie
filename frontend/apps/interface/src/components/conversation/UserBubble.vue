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
    <span class="sender-glyph" role="img" aria-label="You said"></span>

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
