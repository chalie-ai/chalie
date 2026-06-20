import { getHost, api } from '@chalie/shared';

/** Response from GET /voice/health */
export interface VoiceHealth {
  status: 'ok' | 'loading' | 'unavailable';
  reason?: string;
  missing?: string[];
  hint?: string;
}

export const voice = {
  /**
   * GET /voice/health — returns status in {ok, loading, unavailable}.
   * Public probe (no auth, always 200), like /ready and /health, so it opts out
   * of the client's redirect-on-401: a probe failure should mark voice
   * unavailable (handled by the store's catch), never yank the user to /login/.
   */
  health(): Promise<VoiceHealth> {
    return api.get('/voice/health', { redirectOnAuthError: false });
  },

  /**
   * POST /voice/transcribe — multipart FormData with file=recording.wav.
   * Returns raw Response so the caller can handle streaming/retry.
   */
  transcribe(file: File): Promise<Response> {
    const formData = new FormData();
    formData.append('file', file, 'recording.wav');
    return fetch(`${getHost().replace(/\/$/, '')}/voice/transcribe`, {
      method: 'POST',
      credentials: 'same-origin',
      body: formData,
    });
  },

  /**
   * POST /voice/synthesize — body: { text }. Returns raw Response (binary audio/wav blob).
   * HTTP 503 + reason:'loading' means the TTS model is still warming up; the player
   * handles the Retry-After retry loop.
   */
  speak(text: string): Promise<Response> {
    return fetch(`${getHost().replace(/\/$/, '')}/voice/synthesize`, {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
  },
};
