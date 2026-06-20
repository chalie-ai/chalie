import type { PlatformAdapter, WakeLockHandle } from './PlatformAdapter';

export const webPlatformAdapter: PlatformAdapter = {
  getUserMedia: (constraints) =>
    // navigator.mediaDevices is undefined in non-secure (HTTP) contexts; a bare
    // call would throw synchronously and escape callers' .catch(), so guard to
    // keep the always-returns-a-Promise contract.
    navigator.mediaDevices
      ? navigator.mediaDevices.getUserMedia(constraints)
      : Promise.reject(
          new DOMException('getUserMedia unavailable in non-secure context', 'NotSupportedError'),
        ),
  createAudioContext: () => {
    const Ctor =
      globalThis.AudioContext ??
      (globalThis as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    return new Ctor();
  },
  notificationPermission: () => ('Notification' in globalThis ? Notification.permission : 'denied'),
  requestNotificationPermission: () =>
    'Notification' in globalThis
      ? Notification.requestPermission()
      : Promise.resolve('denied' as NotificationPermission),
  showNotification: (title, options) => {
    if ('Notification' in globalThis && Notification.permission === 'granted') {
      new Notification(title, options);
    }
  },
  getCurrentPosition: (options) =>
    new Promise((resolve, reject) =>
      navigator.geolocation.getCurrentPosition(resolve, reject, options),
    ),
  readFileAsDataURL: (file) =>
    new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result as string);
      reader.onerror = () => reject(reader.error);
      reader.readAsDataURL(file);
    }),
  getItem: (key) => {
    try {
      return localStorage.getItem(key);
    } catch {
      return null;
    }
  },
  setItem: (key, value) => {
    try {
      localStorage.setItem(key, value);
    } catch {
      /* ignore */
    }
  },
  removeItem: (key) => {
    try {
      localStorage.removeItem(key);
    } catch {
      /* ignore */
    }
  },
  openBrain: () => {
    globalThis.open('/brain/', '_blank');
  },
  createWakeLock: (): WakeLockHandle => {
    // The Screen Wake Lock API auto-releases when the page becomes hidden
    // (tab-switch, lock-screen), so we re-request on visibilitychange while
    // still active. No-op on browsers without navigator.wakeLock.
    let sentinel: WakeLockSentinel | null = null;
    let active = false;

    const requestLock = async () => {
      if (!('wakeLock' in navigator)) return;
      try {
        sentinel = await (navigator as Navigator & { wakeLock: { request(type: string): Promise<WakeLockSentinel> } }).wakeLock.request('screen');
        sentinel.addEventListener('release', () => { sentinel = null; });
      } catch (err) {
        console.debug('[wakelock] request failed:', err);
      }
    };

    const onVisibility = () => {
      if (active && document.visibilityState === 'visible' && !sentinel) {
        void requestLock();
      }
    };

    return {
      async acquire() {
        if (active) return;
        active = true;
        document.addEventListener('visibilitychange', onVisibility);
        await requestLock();
      },
      async release() {
        if (!active) return;
        active = false;
        document.removeEventListener('visibilitychange', onVisibility);
        if (sentinel) {
          try { await sentinel.release(); } catch { /* ignore */ }
          sentinel = null;
        }
      },
    };
  },
};
