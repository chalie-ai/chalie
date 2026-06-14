<script setup lang="ts">
import { onMounted } from 'vue';
import { useShellStore } from '../../stores/shell';

const shell = useShellStore();
onMounted(() => { shell.subTitle = 'Processing'; });
</script>

<template>
  <div class="panel-placeholder"><p>Documents Processing — implementation in P2c.</p></div>
</template>
