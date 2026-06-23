<script setup lang="ts">
import type { ErrorForm } from '../../stores/conversation';

defineProps<{ form: ErrorForm }>();
</script>

<template>
  <div class="speech-form speech-form--chalie speech-form--error">
    <!-- Glyph + intentionally-empty timestamp span (matches Chalie header layout). -->
    <div class="speech-form__header">
      <span class="sender-glyph" aria-hidden="true"></span>
      <span class="speech-form__timestamp" aria-hidden="true"></span>
    </div>
    <div class="speech-form__text">{{ form.message }}</div>
  </div>
</template>
