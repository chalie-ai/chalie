<script setup lang="ts">
import { onMounted } from 'vue';
import { useShellStore } from '../../stores/shell';

const shell = useShellStore();
onMounted(() => { shell.subTitle = 'All'; });
</script>

<template>
  <div class="panel-placeholder"><p>Scheduler All — implementation in P2c.</p></div>
</template>
