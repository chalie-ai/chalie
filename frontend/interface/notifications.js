/**
 * Audio chime and system notifications.
 */
export class Notifications {
  constructor() {
    this._audioCtx = null;
  }

  /**
   * Creates (or resumes) the AudioContext. Call once on the first user gesture
   * so the autoplay policy is satisfied before any chime is needed.
   */
  unlockAudio() {
    if (!this._audioCtx) {
      this._audioCtx = new (globalThis.AudioContext || globalThis.webkitAudioContext)();
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
        this._audioCtx = new (globalThis.AudioContext || globalThis.webkitAudioContext)();
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
   */
  notifyBackground(text) {
    if (Notification.permission !== 'granted') return;

    const body = text.length > 200 ? text.slice(0, 200) + '…' : text;

    try { new Notification('Chalie', { body, tag: 'chalie-message' }); } catch (e) { console.warn('[notifications] Notification API failed:', e); }

    this.playChime();
  }
}
