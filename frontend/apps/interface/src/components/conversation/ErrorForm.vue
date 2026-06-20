<script setup lang="ts">
import type { ErrorForm } from '../../stores/conversation';

defineProps<{ form: ErrorForm }>();
</script>

<template>
  <div class="speech-form speech-form--chalie speech-form--error">
    <!-- Header row mirroring legacy _buildChalieHeader({}) — glyph + empty timestamp -->
    <div class="speech-form__header">
      <span class="sender-glyph" aria-hidden="true"></span>
      <!-- FIX 9: empty timestamp span after glyph, matching legacy _buildChalieHeader({}) -->
      <span class="speech-form__timestamp" aria-hidden="true"></span>
    </div>
    <div class="speech-form__text">{{ form.message }}</div>
  </div>
</template>
