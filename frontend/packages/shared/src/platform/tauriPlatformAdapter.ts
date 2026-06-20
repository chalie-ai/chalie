import type { PlatformAdapter, WakeLockHandle } from './PlatformAdapter';
import { webPlatformAdapter } from './webPlatformAdapter';
import { getHost } from '../config/host';

/**
 * Tauri (native shell) PlatformAdapter. The webview is a real browser context,
 * so media + storage primitives delegate to the web adapter; only the
 * native-bridged capabilities differ: notifications (@tauri-apps/plugin-
 * notification), external Brain (@tauri-apps/plugin-opener), and on-device STT
 * (custom tauri-plugin-stt, identifier 'stt').
 *
 * The @tauri-apps/* packages are declared as dependencies of @chalie/shared, so
 * the interface app's bundler resolves these dynamic imports into lazy chunks.
 * Those chunks are only fetched inside the Tauri webview — the runtime selector
 * picks this adapter only when window.__TAURI__ is present, so the web build
 * never loads them.
 */
export const tauriPlatformAdapter: PlatformAdapter = {
  // Webview shares the browser primitives — delegate.
  getUserMedia: (constraints) => webPlatformAdapter.getUserMedia(constraints),
  createAudioContext: () => webPlatformAdapter.createAudioContext(),
  getCurrentPosition: (options) => webPlatformAdapter.getCurrentPosition(options),
  readFileAsDataURL: (file) => webPlatformAdapter.readFileAsDataURL(file),
  getItem: (key) => webPlatformAdapter.getItem(key),
  setItem: (key, value) => webPlatformAdapter.setItem(key, value),
  removeItem: (key) => webPlatformAdapter.removeItem(key),
  createWakeLock: (): WakeLockHandle => webPlatformAdapter.createWakeLock(),

  notificationPermission: () =>
    // Synchronous web contract; the native grant is reconciled lazily via
    // requestNotificationPermission, so the cached web value is the gate.
    webPlatformAdapter.notificationPermission(),

  requestNotificationPermission: async (): Promise<NotificationPermission> => {
    const { isPermissionGranted, requestPermission } = await import('@tauri-apps/plugin-notification');
    if (await isPermissionGranted()) return 'granted';
    const result = await requestPermission();
    return result === 'granted' ? 'granted' : 'denied';
  },

  showNotification: (title, options) => {
    void (async () => {
      const { isPermissionGranted, requestPermission, sendNotification } =
        await import('@tauri-apps/plugin-notification');
      let granted = await isPermissionGranted();
      if (!granted) granted = (await requestPermission()) === 'granted';
      if (granted) sendNotification({ title, body: options?.body });
    })();
  },

  openBrain: () => {
    void (async () => {
      const { openUrl } = await import('@tauri-apps/plugin-opener');
      await openUrl(getHost().replace(/\/$/, '') + '/brain');
    })();
  },

  startSTT: async (): Promise<void> => {
    const { invoke } = await import('@tauri-apps/api/core');
    await invoke('plugin:stt|start');
  },
  stopSTT: async (): Promise<void> => {
    const { invoke } = await import('@tauri-apps/api/core');
    await invoke('plugin:stt|stop');
  },
};
