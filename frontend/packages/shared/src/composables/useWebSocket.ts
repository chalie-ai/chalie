import { WebSocketService } from '../services/WebSocketService';
import { getHost } from '../config/host';

let service: WebSocketService | null = null;

/** Process-wide singleton so every view shares one socket. */
export function getWebSocket(): WebSocketService {
  service ??= new WebSocketService(getHost);
  return service;
}
