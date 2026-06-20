<script setup lang="ts">
import { ref, onMounted } from 'vue';
import {
  ApiClient,
  getHost,
  getToken,
  useTheme,
  useWebSocket,
  BaseButton,
  BaseCard,
  BaseField,
} from '@chalie/shared';

const { theme, toggle } = useTheme();
const { connected } = useWebSocket();
const api = new ApiClient(getHost, getToken);
const ready = ref<string>('checking…');
const note = ref('');

onMounted(async () => {
  ready.value = (await api.ready()).ready ? 'ready' : 'not-ready';
});
</script>

<template>
  <main data-testid="home" style="max-width: 640px; margin: 2rem auto; padding: 0 1rem">
    <BaseCard title="Chalie — Vue foundation">
      <p>
        Theme: <strong data-testid="theme">{{ theme }}</strong>
      </p>
      <p>
        WebSocket: <strong data-testid="ws-status">{{ connected ? 'connected' : 'disconnected' }}</strong>
      </p>
      <p>
        Backend: <strong data-testid="ready">{{ ready }}</strong>
      </p>
      <BaseField v-model="note" data-testid="note" label="Scratch note" placeholder="type…" />
      <div style="display: flex; gap: 0.5rem; margin-top: 1rem">
        <BaseButton data-testid="toggle-theme" @click="toggle">Toggle theme</BaseButton>
        <BaseButton variant="ghost">Ghost</BaseButton>
      </div>
    </BaseCard>
  </main>
</template>
