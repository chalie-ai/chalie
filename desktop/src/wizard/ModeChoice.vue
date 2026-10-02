<script setup lang="ts">
const emit = defineEmits<{ (e: 'choose', choice: 'existing' | 'install'): void }>();
</script>

<template>
  <h1>Set up Chalie</h1>
  <p class="lead">Chalie runs on a server. Point this app at one.</p>
  <div class="choices">
    <button type="button" class="primary" @click="emit('choose', 'existing')">
      Connect to an existing Chalie
      <small>It already runs on this Mac, on another machine, or on a server you own.</small>
    </button>
    <button type="button" @click="emit('choose', 'install')">
      Install Chalie on this Mac
      <small>Set up a new Chalie server here.</small>
    </button>
  </div>
</template>
