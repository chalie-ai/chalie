<script setup lang="ts">
import { onMounted } from 'vue';
import { useShellStore } from '../../stores/shell';

const shell = useShellStore();
onMounted(() => { shell.subTitle = 'Compaction'; });
</script>

<template>
  <div class="panel-placeholder"><p>Compaction sub-panel — implementation in P2b.</p></div>
</template>
