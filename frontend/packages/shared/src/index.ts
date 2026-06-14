export { ApiClient, AuthError, HttpError } from './services/ApiClient';
export type { GetHost } from './services/types';
export { WebSocketService } from './services/WebSocketService';
export type {
  WsInboundEvent,
  WsPushEvent,
  WsPushType,
  WsMessageEvent,
  ChatCallbacks,
  ActionCallbacks,
} from './services/WebSocketService';
export { getHost, setHost } from './config/host';
export type { PlatformAdapter, WakeLockHandle } from './platform/PlatformAdapter';
export { webPlatformAdapter } from './platform/webPlatformAdapter';
export { useThemeStore } from './stores/theme';
export type { Theme } from './stores/theme';
export { useConnectionStore } from './stores/connection';
export { useTheme } from './composables/useTheme';
export { useWebSocket, getWebSocket } from './composables/useWebSocket';
export { useApiClient } from './composables/useApiClient';
export { default as BaseButton } from './ui/BaseButton.vue';
export { default as BaseCard } from './ui/BaseCard.vue';
export { default as BaseField } from './ui/BaseField.vue';
export { default as BaseModal } from './ui/BaseModal.vue';
export { default as BaseTooltip } from './ui/BaseTooltip.vue';
