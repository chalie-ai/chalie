<script setup lang="ts">
import { onMounted } from 'vue';
import { useShellStore } from '../../stores/shell';

const shell = useShellStore();
onMounted(() => { shell.subTitle = 'Pending'; });
</script>

<template>
  <div class="panel-placeholder"><p>Scheduler Pending — implementation in P2c.</p></div>
</template>
