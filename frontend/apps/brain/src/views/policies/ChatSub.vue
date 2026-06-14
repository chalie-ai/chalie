<script setup lang="ts">
import { onMounted } from 'vue';
import { useShellStore } from '../../stores/shell';

const shell = useShellStore();
onMounted(() => { shell.subTitle = 'Chat'; });
</script>

<template>
  <div class="panel-placeholder"><p>Policies Chat — implementation in P2d.</p></div>
</template>
