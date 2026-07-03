export { ApiClient, AuthError, HttpError } from './services/ApiClient';
export type { RequestOpts, AuthErrorHandler } from './services/ApiClient';
export type { GetHost } from './services/types';
export { WebSocketService } from './services/WebSocketService';
export type {
  WsInboundEvent,
  WsPushEvent,
  WsPushType,
  WsMessageEvent,
  WsTurnExecutionEvent,
  TurnExecutionState,
  ActionCallbacks,
} from './services/WebSocketService';
export { ConfigType } from './config/configType';
export { getHost, setHost, getToken, setToken, getUsername, setUsername } from './config/host';
export type { PairingPayload } from './config/pairing';
export { validatePairingPayload } from './config/pairing';
export type { PlatformAdapter, WakeLockHandle } from './platform/PlatformAdapter';
export { webPlatformAdapter } from './platform/webPlatformAdapter';
export { tauriPlatformAdapter } from './platform/tauriPlatformAdapter';
export { platform, isTauri } from './platform';
export { useThemeStore } from './stores/theme';
export type { Theme } from './stores/theme';
export { useConnectionStore } from './stores/connection';
export { useTheme } from './composables/useTheme';
export { useWebSocket, getWebSocket } from './composables/useWebSocket';
export { useApiClient, api } from './composables/useApiClient';
export { useAsyncResource } from './composables/useAsyncResource';
export type { AsyncResource, AsyncResourceOptions } from './composables/useAsyncResource';
export { default as BaseButton } from './ui/BaseButton.vue';
export { default as BaseCard } from './ui/BaseCard.vue';
export { default as BaseField } from './ui/BaseField.vue';
export { default as BaseModal } from './ui/BaseModal.vue';
export { default as BaseTooltip } from './ui/BaseTooltip.vue';
