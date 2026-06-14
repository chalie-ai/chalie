<script setup lang="ts">
import { onMounted } from 'vue';
import { useShellStore } from '../stores/shell';

const shell = useShellStore();
onMounted(() => { shell.panelTitle = 'Lists'; shell.subTitle = ''; });
</script>

<template>
  <div class="panel-header"><h2>Lists</h2></div>
  <div class="panel-placeholder"><p>Lists panel — implementation in P2c.</p></div>
</template>
