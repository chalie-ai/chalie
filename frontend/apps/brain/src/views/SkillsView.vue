<script setup lang="ts">
import { onMounted } from 'vue';
import { useShellStore } from '../stores/shell';

const shell = useShellStore();
onMounted(() => { shell.panelTitle = 'Skills'; shell.subTitle = ''; });
</script>

<template>
  <div class="panel-header"><h2>Skills</h2></div>
  <div class="panel-placeholder"><p>Skills panel — implementation in P2e.</p></div>
</template>
