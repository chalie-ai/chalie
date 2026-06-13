import type { PlatformAdapter } from './PlatformAdapter';

export const webPlatformAdapter: PlatformAdapter = {
  getUserMedia: (constraints) =>
    // navigator.mediaDevices is undefined in non-secure (HTTP) contexts; guard so
    // the PlatformAdapter contract (always returns a Promise) holds — a bare call
    // would throw synchronously and escape callers' .catch() handlers.
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
};
