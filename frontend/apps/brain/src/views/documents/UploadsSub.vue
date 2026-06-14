<script setup lang="ts">
import { onMounted } from 'vue';
import { useShellStore } from '../../stores/shell';

const shell = useShellStore();
onMounted(() => { shell.subTitle = 'Uploads'; });
</script>

<template>
  <div class="panel-placeholder"><p>Documents Uploads — implementation in P2c.</p></div>
</template>
