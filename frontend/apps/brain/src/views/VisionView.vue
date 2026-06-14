<script setup lang="ts">
import { onMounted } from 'vue';
import { useShellStore } from '../stores/shell';

const shell = useShellStore();
onMounted(() => { shell.panelTitle = 'Vision'; shell.subTitle = ''; });
</script>

<template>
  <div class="panel-header"><h2>Vision</h2></div>
  <div class="panel-placeholder"><p>Vision panel — implementation in P2b.</p></div>
</template>
