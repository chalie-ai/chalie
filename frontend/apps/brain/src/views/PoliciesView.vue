<script setup lang="ts">
import { RouterView } from 'vue-router';
import { onMounted } from 'vue';
import { useShellStore } from '../stores/shell';

const shell = useShellStore();
onMounted(() => { shell.panelTitle = 'Policies'; });
</script>

<template>
  <div class="panel-header"><h2>Policies</h2></div>
  <RouterView />
</template>
