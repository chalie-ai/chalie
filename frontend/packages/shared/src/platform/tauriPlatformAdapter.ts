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
// Native OS notification grant. Inside a Tauri webview the web Notification API
// stays 'default' even after the OS grants permission, so the synchronous gate
// must read this cache — seeded at startup via requestNotificationPermission and
// refreshed on every native send — rather than the (stale) web value.
let _nativeNotifPerm: NotificationPermission = 'default';
// Unlisten handle for the native STT transcript bridge (Tauri IPC event → DOM bus).
let _sttUnlisten: (() => void) | null = null;

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

  // Sync gate reads the cached native grant (the web API never updates in-webview).
  notificationPermission: () => _nativeNotifPerm,

  requestNotificationPermission: async (): Promise<NotificationPermission> => {
    const { isPermissionGranted, requestPermission } = await import('@tauri-apps/plugin-notification');
    const granted = (await isPermissionGranted()) || (await requestPermission()) === 'granted';
    _nativeNotifPerm = granted ? 'granted' : 'denied';
    return _nativeNotifPerm;
  },

  showNotification: (title, options) => {
    void (async () => {
      const { isPermissionGranted, requestPermission, sendNotification } =
        await import('@tauri-apps/plugin-notification');
      let granted = await isPermissionGranted();
      if (!granted) granted = (await requestPermission()) === 'granted';
      _nativeNotifPerm = granted ? 'granted' : 'denied';
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
    const [{ invoke }, { listen }] = await Promise.all([
      import('@tauri-apps/api/core'),
      import('@tauri-apps/api/event'),
    ]);
    // The plugin streams transcripts over Tauri's IPC event bus; the app's
    // compose-box listener is on the DOM CustomEvent bus, so bridge each across.
    _sttUnlisten?.();
    _sttUnlisten = await listen<{ text: string }>('chalie:voice-transcript', (e) => {
      document.dispatchEvent(
        new CustomEvent('chalie:voice-transcript', { detail: e.payload }),
      );
    });
    await invoke('plugin:stt|start');
  },
  stopSTT: async (): Promise<void> => {
    const { invoke } = await import('@tauri-apps/api/core');
    await invoke('plugin:stt|stop');
    _sttUnlisten?.();
    _sttUnlisten = null;
  },
};
