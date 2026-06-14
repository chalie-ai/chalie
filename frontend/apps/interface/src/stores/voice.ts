/**
 * Voice availability store — availability slice only.
 * Recorder / player slices are wired in P1c.
 *
 * Port of app.js _initVoiceAvailability (lines 310-345): polls GET /voice/health
 * until it terminates, and exposes `available` to gate the mic / speaker controls
 * (InputDock). This is INDEPENDENT of the loading overlay — legacy runs voice
 * availability AFTER the overlay is dismissed (app.js:238 vs :246), so a slow or
 * unavailable voice service must never block first paint. `loading` is purely an
 * internal "still polling" flag; nothing outside this store gates on it.
 */
import { defineStore } from 'pinia';
import { voice } from '../api/voice';

const POLL_INTERVAL_MS = 2000;
const MAX_POLL_MS = 60_000;

export const useVoiceStore = defineStore('voice', {
  state: () => ({
    /** True once /voice/health returns status==='ok'. */
    available: false,
    /** True while the health poll is in flight (mic stays hidden until done). */
    loading: true,
  }),

  actions: {
    /**
     * Poll GET /voice/health until it terminates, then set `available`.
     *
     * status 'ok'          → available=true  (reveal mic), stop
     * status 'unavailable' → available=false (hide mic),  stop
     * status 'loading'     → service warming up; re-poll every 2s up to 60s
     * timeout              → available=false (treat warm-up that never finished
     *                        as unavailable, matching legacy hide())
     * thrown error         → available=false (server unreachable), stop
     *
     * `loading` is cleared in every terminal branch.
     */
    async checkAvailability(): Promise<void> {
      const deadline = Date.now() + MAX_POLL_MS;
      for (;;) {
        try {
          const data = await voice.health();
          if (data.status === 'ok') {
            this.available = true;
            this.loading = false;
            return;
          }
          if (data.status === 'unavailable') {
            this.available = false;
            this.loading = false;
            return;
          }
          // 'loading' — re-poll until the deadline, then treat as unavailable.
          if (Date.now() >= deadline) {
            this.available = false;
            this.loading = false;
            return;
          }
        } catch {
          // Network error or server unreachable — treat as unavailable.
          this.available = false;
          this.loading = false;
          return;
        }
        await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
      }
    },
  },
});
