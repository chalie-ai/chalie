/**
 * The single seam between Chalie's UI and host-platform capabilities.
 * Web ships the only implementation in P0; a later epic adds capacitor/tauri
 * impls so the interface app can be wrapped natively without touching callers.
 */
export interface PlatformAdapter {
  // Audio capture / playback (voice_recorder, voice_player)
  getUserMedia(constraints: MediaStreamConstraints): Promise<MediaStream>;
  createAudioContext(): AudioContext;

  // Notifications (notifications.js)
  notificationPermission(): NotificationPermission;
  requestNotificationPermission(): Promise<NotificationPermission>;
  showNotification(title: string, options?: NotificationOptions): void;

  // Geolocation (heartbeat / ambient)
  getCurrentPosition(options?: PositionOptions): Promise<GeolocationPosition>;

  // Files (image_attach, document_upload)
  readFileAsDataURL(file: File): Promise<string>;

  // Key/value storage
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;

  // Navigation — web opens a tab; native opens an in-app webview (later epic)
  openBrain(): void;

  // Screen Wake Lock — keeps the display on during voice record/playback (C1/C5)
  createWakeLock(): WakeLockHandle;
}

/** Handle returned by PlatformAdapter.createWakeLock(). */
export interface WakeLockHandle {
  /** Request the screen wake lock. No-op when already active. */
  acquire(): Promise<void>;
  /** Release the screen wake lock. No-op when not active. */
  release(): Promise<void>;
}
