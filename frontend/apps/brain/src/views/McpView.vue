<script setup lang="ts">
import { onMounted } from 'vue';
import { useShellStore } from '../stores/shell';

const shell = useShellStore();
onMounted(() => { shell.panelTitle = 'MCP'; shell.subTitle = ''; });
</script>

<template>
  <div class="panel-header"><h2>MCP</h2></div>
  <div class="panel-placeholder"><p>MCP panel — implementation in P2f.</p></div>
</template>
