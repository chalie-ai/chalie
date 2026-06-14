<script setup lang="ts">
import { onMounted } from 'vue';
import { useShellStore } from '../../stores/shell';

const shell = useShellStore();
onMounted(() => { shell.subTitle = 'Deleted'; });
</script>

<template>
  <div class="panel-placeholder"><p>Documents Deleted — implementation in P2c.</p></div>
</template>
