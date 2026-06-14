/**
 * Voice availability store — availability slice only.
 * Recorder / player slices are wired in P1c.
 *
 * Port of app.js _initVoiceAvailability: maps voice.health() status to
 * available + loading flags used by LoadingOverlay (gate) and InputDock
 * (mic visibility).
 */
import { defineStore } from 'pinia';
import { voice } from '../api/voice';

export const useVoiceStore = defineStore('voice', {
  state: () => ({
    /** True once /voice/health returns status==='ok'. */
    available: false,
    /** True until the health check resolves (ok, unavailable, or error). */
    loading: true,
  }),

  actions: {
    /**
     * Call GET /voice/health and update available + loading.
     *
     * status 'ok'          → available=true,  loading=false
     * status 'loading'     → available=false, loading=true  (still in progress)
     * status 'unavailable' → available=false, loading=false
     * thrown error         → available=false, loading=false
     */
    async checkAvailability(): Promise<void> {
      try {
        const data = await voice.health();
        if (data.status === 'ok') {
          this.available = true;
          this.loading = false;
        } else if (data.status === 'loading') {
          // Voice service is still initialising; keep loading=true so the
          // overlay waits, then stay available=false until a re-check (P1c)
          // resolves to 'ok'.
          this.available = false;
          this.loading = true;
        } else {
          // 'unavailable' or any unrecognised status
          this.available = false;
          this.loading = false;
        }
      } catch {
        // Network error or server unreachable — treat as unavailable
        this.available = false;
        this.loading = false;
      }
    },
  },
});
