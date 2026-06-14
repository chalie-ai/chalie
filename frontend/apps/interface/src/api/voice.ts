import { getHost } from '@chalie/shared';
import { useApiClient } from '@chalie/shared';

/** Response from GET /voice/health */
export interface VoiceHealth {
  status: 'ok' | 'loading' | 'unavailable';
  reason?: string;
  missing?: string[];
  hint?: string;
}

export const voice = {
  /** GET /voice/health — returns status in {ok, loading, unavailable}. */
  health(): Promise<VoiceHealth> {
    const api = useApiClient();
    return api.get('/voice/health');
  },

  /**
   * POST /voice/transcribe — multipart FormData with file=recording.wav.
   * Returns raw Response so the caller can handle streaming/retry.
   */
  transcribe(file: File): Promise<Response> {
    const host = getHost();
    const base = host ? host.replace(/\/$/, '') : '';
    const formData = new FormData();
    formData.append('file', file, 'recording.wav');
    return fetch(`${base}/voice/transcribe`, {
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
    const host = getHost();
    const base = host ? host.replace(/\/$/, '') : '';
    return fetch(`${base}/voice/synthesize`, {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
  },
};
