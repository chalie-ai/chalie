/**
 * Notifications store — audio chime and OS notifications.
 *
 * Browser-API access goes exclusively through the runtime platform adapter (no raw
 * window.Notification / localStorage / new AudioContext()); HTTP through api
 * wrappers only.
 */
import { defineStore } from 'pinia';
import { platform as adapter } from '@chalie/shared';

const CHIME_FREQ_HZ = 880;        // A5
const CHIME_DURATION_S = 0.5;
const CHIME_GAIN_START = 0.3;
const CHIME_GAIN_END = 0.01;

const NOTIFY_TITLE = 'Chalie';
const NOTIFY_BODY_MAX = 200;
const NOTIFY_TAG = 'chalie-message';
const MAX_NOTIFICATIONS = 50;

export interface NotificationItem {
  id?: string;
  [key: string]: unknown;
}

// Singleton AudioContext — created lazily, shared across chimes.
let _audioCtx: AudioContext | null = null;

function getAudioContext(): AudioContext {
  _audioCtx ??= adapter.createAudioContext();
  return _audioCtx;
}

function playChime(): void {
  try {
    const ctx = getAudioContext();
    if ((ctx.state as string) === 'suspended') {
      void ctx.resume();
    }
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.frequency.value = CHIME_FREQ_HZ;
    osc.type = 'sine';
    gain.gain.setValueAtTime(CHIME_GAIN_START, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(CHIME_GAIN_END, ctx.currentTime + CHIME_DURATION_S);
    osc.start(ctx.currentTime);
    osc.stop(ctx.currentTime + CHIME_DURATION_S);
  } catch (e) {
    console.warn('[notifications] AudioContext unavailable or blocked:', e);
  }
}

function showOsNotification(text: string): void {
  if (adapter.notificationPermission() !== 'granted') return;
  const body = text.length > NOTIFY_BODY_MAX ? text.slice(0, NOTIFY_BODY_MAX) + '…' : text;
  try {
    adapter.showNotification(NOTIFY_TITLE, { body, tag: NOTIFY_TAG });
  } catch (e) {
    console.warn('[notifications] Notification API failed:', e);
  }
}

export const useNotificationsStore = defineStore('notifications', {
  state: () => ({
    notifications: [] as NotificationItem[],
  }),

  actions: {
    /** Resume the AudioContext on a user gesture to satisfy autoplay policy. */
    unlockAudio(): void {
      try {
        const ctx = getAudioContext();
        if ((ctx.state as string) === 'suspended') {
          void ctx.resume();
        }
      } catch {
        // Non-fatal — playChime will handle its own error.
      }
    },

    /**
     * Background notification (chime + OS) when a message arrives while the tab
     * is blurred. Both the chime AND the OS notification are gated behind
     * notification permission — ungranted = whole method is a no-op.
     */
    pushBackground(text: string): void {
      if (!text) return;
      if (adapter.notificationPermission() !== 'granted') return;
      showOsNotification(text);
      playChime();
      this.notifications.push({ id: String(Date.now()), text });
      if (this.notifications.length > MAX_NOTIFICATIONS) {
        this.notifications.splice(0, this.notifications.length - MAX_NOTIFICATIONS);
      }
    },
  },
});
