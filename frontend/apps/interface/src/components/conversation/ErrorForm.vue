<script setup lang="ts">
import type { ErrorForm } from '../../stores/conversation';

defineProps<{ form: ErrorForm }>();
</script>

<template>
  <div class="speech-form speech-form--chalie speech-form--error">
    <!-- Header row mirroring legacy replaceActWithError structure -->
    <div class="speech-form__header">
      <svg class="sender-glyph" viewBox="0 0 18 18" fill="none" aria-hidden="true">
        <line x1="4" y1="14" x2="9" y2="5" stroke="currentColor" stroke-width="0.6" opacity="0.45" />
        <line x1="9" y1="5" x2="14" y2="11" stroke="currentColor" stroke-width="0.6" opacity="0.45" />
        <line x1="4" y1="14" x2="14" y2="11" stroke="currentColor" stroke-width="0.6" opacity="0.45" />
        <circle class="sender-glyph__dot" cx="4" cy="14" r="1.4" fill="currentColor" />
        <circle class="sender-glyph__dot" cx="9" cy="5" r="1.6" fill="currentColor" />
        <circle class="sender-glyph__dot" cx="14" cy="11" r="1.3" fill="currentColor" />
        <circle class="sender-glyph__dot" cx="11" cy="2" r="0.9" fill="currentColor" opacity="0.65" />
      </svg>
    </div>
    <div class="speech-form__text">{{ form.message }}</div>
  </div>
</template>
