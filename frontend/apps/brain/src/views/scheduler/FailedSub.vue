<script setup lang="ts">
import { onMounted } from 'vue';
import { useShellStore } from '../../stores/shell';

const shell = useShellStore();
onMounted(() => { shell.subTitle = 'Failed'; });
</script>

<template>
  <div class="panel-placeholder"><p>Scheduler Failed — implementation in P2c.</p></div>
</template>
