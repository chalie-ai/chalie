import { onMounted } from 'vue';
import { storeToRefs } from 'pinia';
import { WebSocketService } from '../services/WebSocketService';
import { getHost } from '../config/host';
import { useConnectionStore } from '../stores/connection';

let service: WebSocketService | null = null;
let focusBound = false;

/** Process-wide singleton so every view shares one socket. */
export function getWebSocket(): WebSocketService {
  if (!service) service = new WebSocketService(getHost);
  return service;
}

export function useWebSocket() {
  const ws = getWebSocket();
  const conn = useConnectionStore();
  const { connected } = storeToRefs(conn);

  onMounted(() => {
    ws.onDisconnect(() => conn.setConnected(false));
    ws.onAny(() => conn.setConnected(ws.isConnected));
    ws.onConnect(() => conn.setConnected(ws.isConnected));
    ws.connect();
    conn.setConnected(ws.isConnected);
    if (!focusBound) {
      focusBound = true;
      globalThis.addEventListener('focus', () => ws.ensureAlive());
    }
  });

  return { ws, connected };
}
