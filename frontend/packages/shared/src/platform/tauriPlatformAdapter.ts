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
 * The @tauri-apps/* and @tauri-apps/api packages are added in the native-shell
 * phase; until then the lazy `import(...)` specifiers carry `@vite-ignore` so
 * the web build never pulls them in, and `@ts-expect-error` so tsc accepts the
 * not-yet-present module specifiers.
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
    // @ts-expect-error tauri plugin added in the native-shell phase
    const { isPermissionGranted, requestPermission } = await import(/* @vite-ignore */ '@tauri-apps/plugin-notification');
    if (await isPermissionGranted()) return 'granted';
    const result = await requestPermission();
    return result === 'granted' ? 'granted' : 'denied';
  },

  showNotification: (title, options) => {
    void (async () => {
      const { isPermissionGranted, requestPermission, sendNotification } =
        // @ts-expect-error tauri plugin added in the native-shell phase
        await import(/* @vite-ignore */ '@tauri-apps/plugin-notification');
      let granted = await isPermissionGranted();
      if (!granted) granted = (await requestPermission()) === 'granted';
      if (granted) sendNotification({ title, body: options?.body });
    })();
  },

  openBrain: () => {
    void (async () => {
      // @ts-expect-error tauri plugin added in the native-shell phase
      const { openUrl } = await import(/* @vite-ignore */ '@tauri-apps/plugin-opener');
      await openUrl(getHost() + '/brain');
    })();
  },

  startSTT: async (): Promise<void> => {
    // @ts-expect-error @tauri-apps/api added in the native-shell phase
    const { invoke } = await import(/* @vite-ignore */ '@tauri-apps/api/core');
    await invoke('plugin:stt|start');
  },
  stopSTT: async (): Promise<void> => {
    // @ts-expect-error @tauri-apps/api added in the native-shell phase
    const { invoke } = await import(/* @vite-ignore */ '@tauri-apps/api/core');
    await invoke('plugin:stt|stop');
  },
};
