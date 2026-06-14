<script setup lang="ts">
import { onMounted } from 'vue';
import { useShellStore } from '../../stores/shell';

const shell = useShellStore();
onMounted(() => { shell.subTitle = 'Cancelled'; });
</script>

<template>
  <div class="panel-placeholder"><p>Scheduler Cancelled — implementation in P2c.</p></div>
</template>
