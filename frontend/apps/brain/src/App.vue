<script setup lang="ts">
import { onMounted } from 'vue';
import { useTheme } from '@chalie/shared';
const { init } = useTheme();
onMounted(() => init());
</script>

<template>
  <main style="padding: 2rem">
    <h1>Brain — Vue scaffold</h1>
    <p>Panels migrate in P2 (TKT-954).</p>
  </main>
</template>
