/**
 * Notifications store — audio chime, OS notifications, tip card, update prompt.
 *
 * Browser-API access goes exclusively through the runtime platform adapter (no raw
 * window.Notification / localStorage / new AudioContext()); HTTP through api
 * wrappers only.
 *
 * QuickTipCard and UpdatePrompt are DORMANT — the backend does not currently
 * emit `quick_tip` or `app_update` WS events — but the state is kept so they
 * activate automatically when the backend ships.
 */
import { defineStore } from 'pinia';
import { platform as adapter } from '@chalie/shared';
import { tips } from '../api/tips';
import { system } from '../api/system';

const CHIME_FREQ_HZ = 880;        // A5
const CHIME_DURATION_S = 0.5;
const CHIME_GAIN_START = 0.3;
const CHIME_GAIN_END = 0.01;

const NOTIFY_TITLE = 'Chalie';
const NOTIFY_BODY_MAX = 200;
const NOTIFY_TAG = 'chalie-message';
const MAX_NOTIFICATIONS = 50;

const LS_UPDATE_DISMISSED = 'chalie_update_dismissed';

export interface NotificationItem {
  id?: string;
  [key: string]: unknown;
}

/** Shape of a quick_tip WS payload (dormant — backend does not yet emit this). */
export interface TipState {
  tip_id: string;
  title?: string;
  body: string;
  example?: string;
  icon_svg?: string;
  category?: string;
}

/** Shape of an app_update WS payload (dormant — backend does not yet emit this). */
export interface UpdateState {
  latest_tag: string;
  latest_version: string;
  current_version: string;
  release_notes?: string;
  /** 'docker' | 'dev' | 'installer' | string */
  deployment_mode?: string;
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
    /** Current quick tip; no-stacking: a new tip replaces the existing one. */
    currentTip: null as TipState | null,
    /** Currently available update. null when none or dismissed. */
    currentUpdate: null as UpdateState | null,
    updateApplyMessage: null as string | null,
    applyingUpdate: false,
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

    /**
     * Chime for a scheduler 'notification' WS event (reminder/task done) —
     * UNCONDITIONAL, no focus or permission gate, unlike pushBackground.
     */
    chime(): void {
      playChime();
    },

    /** No-stacking: replaces any existing tip. DORMANT — backend doesn't emit yet. */
    handleTip(payload: TipState): void {
      if (!payload || !payload.tip_id) return;
      this.currentTip = payload;
    },

    dismissTip(): void {
      const tip = this.currentTip;
      if (!tip) return;
      this.currentTip = null;
      tips.dismiss(tip.tip_id).catch((e: unknown) => {
        console.warn('[notifications] tips.dismiss failed:', e);
      });
    },

    muteTip(): void {
      if (!this.currentTip) return;
      this.currentTip = null;
      tips.mute().catch((e: unknown) => {
        console.warn('[notifications] tips.mute failed:', e);
      });
    },

    /** Skips display if this version was already dismissed. DORMANT. */
    handleUpdate(payload: UpdateState): void {
      if (!payload || !payload.latest_tag) return;
      const dismissed = adapter.getItem(LS_UPDATE_DISMISSED);
      if (dismissed === payload.latest_tag) return;
      this.currentUpdate = payload;
      this.updateApplyMessage = null;
      this.applyingUpdate = false;
    },

    /** Persists the dismissed version tag so it won't reappear. */
    dismissUpdate(): void {
      if (!this.currentUpdate) return;
      adapter.setItem(LS_UPDATE_DISMISSED, this.currentUpdate.latest_tag);
      this.currentUpdate = null;
      this.updateApplyMessage = null;
    },

    async applyUpdate(): Promise<void> {
      if (!this.currentUpdate || this.applyingUpdate) return;
      this.applyingUpdate = true;
      this.updateApplyMessage = null;
      try {
        const result = await system.updateApply(this.currentUpdate.latest_tag);
        if (result.ok) {
          this.updateApplyMessage = result.message ?? 'Restarting Chalie...';
        } else {
          this.updateApplyMessage = result.message || 'Update failed.';
          // Restore actions after 3 s on failure.
          setTimeout(() => {
            this.applyingUpdate = false;
            this.updateApplyMessage = null;
          }, 3000);
        }
      } catch {
        this.updateApplyMessage = 'Update request failed.';
        this.applyingUpdate = false;
      }
    },
  },
});
