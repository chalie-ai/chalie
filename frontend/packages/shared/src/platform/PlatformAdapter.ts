/**
 * The single seam between Chalie's UI and host-platform capabilities, so a host
 * with different primitives can be added without touching callers. The web
 * adapter is the only implementation — the desktop app runs this same frontend.
 */
export interface PlatformAdapter {
  getUserMedia(constraints: MediaStreamConstraints): Promise<MediaStream>;
  createAudioContext(): AudioContext;

  notificationPermission(): NotificationPermission;
  requestNotificationPermission(): Promise<NotificationPermission>;
  showNotification(title: string, options?: NotificationOptions): void;

  getCurrentPosition(options?: PositionOptions): Promise<GeolocationPosition>;

  readFileAsDataURL(file: File): Promise<string>;

  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;

  // opens the Brain dashboard (a new tab on the web adapter)
  openBrain(): void;

  // keeps the display on during voice record/playback
  createWakeLock(): WakeLockHandle;
}

export interface WakeLockHandle {
  /** Request the screen wake lock. No-op when already active. */
  acquire(): Promise<void>;
  /** Release the screen wake lock. No-op when not active. */
  release(): Promise<void>;
}
