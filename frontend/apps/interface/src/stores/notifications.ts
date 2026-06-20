/**
 * Notifications store — audio chime, OS notifications, tip card, update prompt.
 *
 * Port of:
 *   frontend/interface/notifications.js  (chime + OS notification)
 *   frontend/interface/quick_tip_card.js (tip state)
 *   frontend/interface/update_system.js  (update state)
 *
 * Browser-API access is exclusively through webPlatformAdapter (Rule: no raw
 * window.Notification / localStorage / new AudioContext()).
 * HTTP calls go through api wrappers only (Rule: tips.dismiss / tips.mute).
 *
 * QuickTipCard and UpdatePrompt are DORMANT — the backend does not currently
 * emit `quick_tip` or `app_update` WS events — but the state is ported
 * faithfully so they activate automatically when the backend ships.
 */
import { defineStore } from 'pinia';
import { webPlatformAdapter } from '@chalie/shared';
import { tips } from '../api/tips';
import { system } from '../api/system';

// ── Chime constants (port of notifications.js lines 36-41) ────────────────────
const CHIME_FREQ_HZ = 880;        // A5
const CHIME_DURATION_S = 0.5;
const CHIME_GAIN_START = 0.3;
const CHIME_GAIN_END = 0.01;

// ── OS notification constants (port of notifications.js lines 53-55) ─────────
const NOTIFY_TITLE = 'Chalie';
const NOTIFY_BODY_MAX = 200;
const NOTIFY_TAG = 'chalie-message';

// ── localStorage key for dismissed update version ────────────────────────────
const LS_UPDATE_DISMISSED = 'chalie_update_dismissed';

// ── Types ─────────────────────────────────────────────────────────────────────

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

// ── Singleton AudioContext — created lazily, shared across chimes ─────────────
let _audioCtx: AudioContext | null = null;

function getAudioContext(): AudioContext {
  if (!_audioCtx) {
    _audioCtx = webPlatformAdapter.createAudioContext();
  }
  return _audioCtx;
}

// ── Chime (port of notifications.js playChime, lines 25-45) ──────────────────

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

// ── OS notification (port of notifications.js notifyBackground, lines 50-58) ──

function showOsNotification(text: string): void {
  if (webPlatformAdapter.notificationPermission() !== 'granted') return;
  const body = text.length > NOTIFY_BODY_MAX ? text.slice(0, NOTIFY_BODY_MAX) + '…' : text;
  try {
    webPlatformAdapter.showNotification(NOTIFY_TITLE, { body, tag: NOTIFY_TAG });
  } catch (e) {
    console.warn('[notifications] Notification API failed:', e);
  }
}

// ── Store ─────────────────────────────────────────────────────────────────────

export const useNotificationsStore = defineStore('notifications', {
  state: () => ({
    /** In-app notification list (background events). */
    notifications: [] as NotificationItem[],

    /** Currently displayed quick tip. null when none is showing.
     *  No-stacking: a new tip replaces the existing one (port of quick_tip_card.js:36). */
    currentTip: null as TipState | null,

    /** Currently available update. null when none or dismissed. */
    currentUpdate: null as UpdateState | null,

    /** Result message from the last update/apply call. */
    updateApplyMessage: null as string | null,

    /** True while an update is being applied (shows progress UI). */
    applyingUpdate: false,
  }),

  actions: {
    // ── Audio unlock (call once on first user gesture) ────────────────────────

    /**
     * Unlocks / resumes the AudioContext after a user gesture so the autoplay
     * policy is satisfied before the first chime is needed.
     * Port of notifications.js unlockAudio().
     */
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

    // ── Background notification (chime + OS) ──────────────────────────────────

    /**
     * Show a background notification: play the A5 chime and fire an OS
     * notification (if permission granted).
     * Port of notifications.js notifyBackground() — called by the session store
     * when a message arrives while the tab is not focused.
     */
    pushBackground(text: string): void {
      // Port of notifications.js notifyBackground (50-58): the OS notification
      // AND the chime are gated behind notification permission — when it is not
      // granted the whole method is a no-op (legacy line 51). This is the
      // focus-gated path (session store only calls it while the tab is blurred).
      if (!text) return;
      if (webPlatformAdapter.notificationPermission() !== 'granted') return;
      showOsNotification(text);
      playChime();
      this.notifications.push({ id: String(Date.now()), text });
    },

    // ── Scheduler notification chime ──────────────────────────────────────────

    /**
     * Play the A5 chime for a scheduler 'notification' WS event (reminder fired,
     * task done). Port of app.js onNotification (385-388): chimes UNCONDITIONALLY
     * — no focus or permission gate, unlike the background path above.
     */
    chime(): void {
      playChime();
    },

    // ── Quick tip ─────────────────────────────────────────────────────────────

    /**
     * Handle a `quick_tip` WS push event.
     * No-stacking: replaces any existing tip (port of quick_tip_card.js:36-38).
     * DORMANT — backend does not currently emit this event.
     */
    handleTip(payload: TipState): void {
      if (!payload || !payload.tip_id) return;
      this.currentTip = payload;
    },

    /**
     * Dismiss the current tip (✕ button or Escape key).
     * Sends POST /api/tips/dismiss with { tip_id } to match legacy body shape.
     */
    dismissTip(): void {
      const tip = this.currentTip;
      if (!tip) return;
      this.currentTip = null;
      tips.dismiss(tip.tip_id).catch((e: unknown) => {
        console.warn('[notifications] tips.dismiss failed:', e);
      });
    },

    /**
     * Mute all tips ("Don't show more tips" button).
     * Sends POST /api/tips/mute with empty body to match legacy shape.
     */
    muteTip(): void {
      if (!this.currentTip) return;
      this.currentTip = null;
      tips.mute().catch((e: unknown) => {
        console.warn('[notifications] tips.mute failed:', e);
      });
    },

    // ── Update prompt ─────────────────────────────────────────────────────────

    /**
     * Handle an `app_update` WS push event.
     * Skips display if this version was already dismissed.
     * Port of update_system.js handleUpdateEvent().
     * DORMANT — backend does not currently emit this event.
     */
    handleUpdate(payload: UpdateState): void {
      if (!payload || !payload.latest_tag) return;
      const dismissed = webPlatformAdapter.getItem(LS_UPDATE_DISMISSED);
      if (dismissed === payload.latest_tag) return;
      this.currentUpdate = payload;
      this.updateApplyMessage = null;
      this.applyingUpdate = false;
    },

    /**
     * Dismiss the update banner/dialog without applying.
     * Persists the dismissed version tag so it won't reappear.
     * Port of update_system.js _dismissUpdateBanner().
     */
    dismissUpdate(): void {
      if (!this.currentUpdate) return;
      webPlatformAdapter.setItem(LS_UPDATE_DISMISSED, this.currentUpdate.latest_tag);
      this.currentUpdate = null;
      this.updateApplyMessage = null;
    },

    /**
     * Apply the pending update via POST /system/update/apply { tag }.
     * Shows progress while pending; stores the result message from the response.
     * Port of update_system.js _applyUpdate() / _handleApplyResult().
     */
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
          // Restore actions after 3 s on failure (port of _handleApplyResult lines 136-139).
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
