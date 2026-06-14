<script setup lang="ts">
import { onMounted } from 'vue';
import { useShellStore } from '../../stores/shell';

const shell = useShellStore();
onMounted(() => { shell.subTitle = 'World'; });
</script>

<template>
  <div class="panel-placeholder"><p>World sub-panel — implementation in P2b.</p></div>
</template>
