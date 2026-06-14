<script setup lang="ts">
import { onMounted } from 'vue';
import { useShellStore } from '../stores/shell';

const shell = useShellStore();
onMounted(() => { shell.panelTitle = 'Capabilities'; shell.subTitle = ''; });
</script>

<template>
  <div class="panel-header"><h2>Capabilities</h2></div>
  <div class="panel-placeholder"><p>Capabilities panel — implementation in P2d.</p></div>
</template>
