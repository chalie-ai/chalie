<script setup lang="ts">
import { onMounted } from 'vue';
import { useShellStore } from '../stores/shell';

const shell = useShellStore();
onMounted(() => { shell.panelTitle = 'Providers'; shell.subTitle = ''; });
</script>

<template>
  <div class="panel-header">
    <h2>Providers</h2>
  </div>
  <div class="panel-placeholder">
    <p>Providers panel — implementation in P2b.</p>
  </div>
</template>
