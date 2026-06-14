<script setup lang="ts">
import { RouterView } from 'vue-router';
import { onMounted } from 'vue';
import { useShellStore } from '../stores/shell';

const shell = useShellStore();
onMounted(() => { shell.panelTitle = 'Scheduler'; });
</script>

<template>
  <div class="panel-header"><h2>Scheduler</h2></div>
  <RouterView />
</template>
