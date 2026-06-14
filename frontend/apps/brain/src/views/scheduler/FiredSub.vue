<script setup lang="ts">
import { onMounted } from 'vue';
import { useShellStore } from '../../stores/shell';

const shell = useShellStore();
onMounted(() => { shell.subTitle = 'Fired'; });
</script>

<template>
  <div class="panel-placeholder"><p>Scheduler Fired — implementation in P2c.</p></div>
</template>
