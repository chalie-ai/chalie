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
   * GET /voice/health — public probe (no auth, always 200), so it opts out of
   * redirect-on-401: a failure should mark voice unavailable, never yank the
   * user to /login/.
   */
  health(): Promise<VoiceHealth> {
    return api.get('/api/voice/health', { redirectOnAuthError: false });
  },

  /**
   * POST /voice/transcribe — multipart FormData (file=recording.wav). Returns
   * the raw Response so the caller can handle streaming/retry.
   */
  transcribe(file: File): Promise<Response> {
    const formData = new FormData();
    formData.append('file', file, 'recording.wav');
    return fetch(`${getHost().replace(/\/$/, '')}/api/voice/transcribe`, {
      method: 'POST',
      credentials: 'same-origin',
      body: formData,
    });
  },

  /**
   * POST /voice/synthesize — body { text }, returns raw Response (audio/wav).
   * HTTP 503 + reason:'loading' means the TTS model is warming up; the player
   * runs the Retry-After loop.
   */
  speak(text: string): Promise<Response> {
    return fetch(`${getHost().replace(/\/$/, '')}/api/voice/synthesize`, {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
  },
};
