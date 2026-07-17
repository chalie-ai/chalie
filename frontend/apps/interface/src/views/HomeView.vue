<script setup lang="ts">
import { onMounted, ref } from 'vue';
import {
  ApiClient,
  BaseButton,
  BaseCard,
  BaseField,
  getHost,
  getToken,
  useTheme,
  useWebSocket,
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
  <main data-testid="home" class="page-container">
    <BaseCard title="Chalie — Vue foundation">
      <p>
        Theme: <strong data-testid="theme">{{ theme }}</strong>
      </p>
      <p>
        WebSocket:
        <strong data-testid="ws-status">{{ connected ? 'connected' : 'disconnected' }}</strong>
      </p>
      <p>
        Backend: <strong data-testid="ready">{{ ready }}</strong>
      </p>
      <BaseField v-model="note" data-testid="note" label="Scratch note" placeholder="type…" />
      <div class="action-row">
        <BaseButton data-testid="toggle-theme" @click="toggle">Toggle theme</BaseButton>
        <BaseButton variant="ghost">Ghost</BaseButton>
      </div>
    </BaseCard>
  </main>
</template>
