/**
 * The single seam between Chalie's UI and host-platform capabilities, so the
 * interface app can later be wrapped natively (capacitor/tauri) without touching
 * callers. Web is the only implementation today.
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

  // web opens a tab; native opens an in-app webview (later epic)
  openBrain(): void;

  // keeps the display on during voice record/playback
  createWakeLock(): WakeLockHandle;

  // native on-device streaming STT (final result arrives via the
  // 'chalie:voice-transcript' document event). web rejects 'STT_UNSUPPORTED'
  // so callers fall back to MediaRecorder -> POST /voice/transcribe.
  startSTT(): Promise<void>;
  stopSTT(): Promise<void>;
}

export interface WakeLockHandle {
  /** Request the screen wake lock. No-op when already active. */
  acquire(): Promise<void>;
  /** Release the screen wake lock. No-op when not active. */
  release(): Promise<void>;
}
