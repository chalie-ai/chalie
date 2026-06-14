<script setup lang="ts">
import { onMounted } from 'vue';
import { useShellStore } from '../../stores/shell';

const shell = useShellStore();
onMounted(() => { shell.subTitle = 'External'; });
</script>

<template>
  <div class="panel-placeholder"><p>Policies External — implementation in P2d.</p></div>
</template>
