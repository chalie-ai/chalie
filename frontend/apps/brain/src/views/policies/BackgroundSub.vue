<script setup lang="ts">
import { onMounted } from 'vue';
import { useShellStore } from '../../stores/shell';

const shell = useShellStore();
onMounted(() => { shell.subTitle = 'Background'; });
</script>

<template>
  <div class="panel-placeholder"><p>Policies Background — implementation in P2d.</p></div>
</template>
