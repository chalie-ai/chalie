<script setup lang="ts">
import { onMounted } from 'vue';
import { useShellStore } from '../stores/shell';

const shell = useShellStore();
onMounted(() => { shell.panelTitle = 'Brain'; shell.subTitle = ''; });
</script>

<template>
  <div class="panel-header"><h2>Brain</h2></div>
  <div class="panel-placeholder"><p>Brain panel — implementation in P2f.</p></div>
</template>
