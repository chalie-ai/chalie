<script setup lang="ts">
import { onMounted } from 'vue';
import { useTheme } from '@chalie/shared';

const { init } = useTheme();
onMounted(() => init());
</script>

<template>
  <RouterView />
</template>
