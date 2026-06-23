<script setup lang="ts">
import { ref, onMounted, onBeforeUnmount } from 'vue';
import { useAmbientCanvas } from '../../composables/useAmbientCanvas';

const canvasRef = ref<HTMLCanvasElement | null>(null);
let stop: (() => void) | null = null;

onMounted(() => {
  stop = useAmbientCanvas(canvasRef);
});

onBeforeUnmount(() => {
  stop?.();
});
</script>

<template>
  <canvas id="ambientCanvas" ref="canvasRef"></canvas>
</template>
