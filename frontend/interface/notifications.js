/**
 * Audio chime, system notifications, and push subscription.
 */
export class Notifications {
  constructor({ getHost }) {
    this._getHost = getHost;
    this._audioCtx = null;
  }

  /**
   * Creates (or resumes) the AudioContext. Call once on the first user gesture
   * so the autoplay policy is satisfied before any chime is needed.
   */
  unlockAudio() {
    if (!this._audioCtx) {
      this._audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (this._audioCtx.state === 'suspended') {
      this._audioCtx.resume();
    }
  }

  /**
   * Plays an A5 (880 Hz) sine wave with a 0.5 s exponential decay envelope.
   */
  playChime() {
    try {
      if (!this._audioCtx) {
        this._audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      }
      const ctx = this._audioCtx;
      if (ctx.state === 'suspended') ctx.resume();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.frequency.value = 880; // A5
      osc.type = 'sine';
      gain.gain.setValueAtTime(0.3, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.5);
      osc.start(ctx.currentTime);
      osc.stop(ctx.currentTime + 0.5);
    } catch (e) {
      console.warn('[notifications] AudioContext unavailable or blocked:', e);
    }
  }

  /**
   * Show system notification + play sound when the tab is not focused.
   * Uses ServiceWorkerRegistration.showNotification() so the tag deduplicates
   * against push notifications from sw.js (both use 'chalie-message').
   */
  notifyBackground(text) {
    if (Notification.permission !== 'granted') return;

    const body = text.length > 200 ? text.slice(0, 200) + '…' : text;

    // System notification via SW registration (shared tag prevents duplicates with push)
    if (navigator.serviceWorker?.controller) {
      navigator.serviceWorker.ready.then(reg => {
        reg.showNotification('Chalie', {
          body,
          tag: 'chalie-message',
          data: { url: '/' },
        });
      }).catch(() => {});
    } else {
      // Fallback: Notification API directly (no SW available)
      try { new Notification('Chalie', { body, tag: 'chalie-message' }); } catch (e) { console.warn('[notifications] Notification API failed:', e); }
    }

    // Audible chime — Web Audio may be throttled in hidden tabs but works
    // when the window is just unfocused (another app in foreground).
    this.playChime();
  }

  /**
   * Requests notification permission and subscribes to push via VAPID.
   */
  async requestPushSubscription() {
    if (!('serviceWorker' in navigator) || !('PushManager' in window)) return;

    try {
      const reg = await navigator.serviceWorker.ready;

      // Request notification permission
      const permission = await Notification.requestPermission();
      if (permission !== 'granted') return;

      // Get VAPID public key from backend
      const host = this._getHost();
      const vapidUrl = host
        ? host.replace(/\/$/, '') + '/push/vapid-key'
        : '/push/vapid-key';
      const res = await fetch(vapidUrl);
      if (!res.ok) return;
      const { publicKey } = await res.json();

      // Convert URL-safe base64 to Uint8Array
      const applicationServerKey = this._urlBase64ToUint8Array(publicKey);

      // Subscribe (or get existing subscription)
      let subscription = await reg.pushManager.getSubscription();
      if (!subscription) {
        subscription = await reg.pushManager.subscribe({
          userVisibleOnly: true,
          applicationServerKey,
        });
      }

      // Send subscription to backend
      const subscribeUrl = host
        ? host.replace(/\/$/, '') + '/push/subscribe'
        : '/push/subscribe';
      await fetch(subscribeUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify(subscription.toJSON()),
      });
    } catch (err) {
      console.warn('Push subscription failed:', err);
    }
  }

  /**
   * Clears SW-shown notifications with the 'chalie-message' tag.
   * Call when the user returns to the tab so stale notifications are dismissed.
   */
  dismissNotifications() {
    if (!navigator.serviceWorker?.controller) return;
    navigator.serviceWorker.ready.then(reg => {
      reg.getNotifications({ tag: 'chalie-message' }).then(notes => {
        notes.forEach(n => n.close());
      });
    }).catch(() => {});
  }

  // ---------------------------------------------------------------------------
  // Private helpers
  // ---------------------------------------------------------------------------

  _urlBase64ToUint8Array(base64String) {
    const padding = '='.repeat((4 - base64String.length % 4) % 4);
    const base64 = (base64String + padding).replaceAll('-', '+').replaceAll('_', '/');
    const raw = atob(base64);
    const output = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) {
      output[i] = raw.charCodeAt(i);
    }
    return output;
  }
}
