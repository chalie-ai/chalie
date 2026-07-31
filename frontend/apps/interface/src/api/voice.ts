import { api, getHost } from '@chalie/shared';

/** Response from GET /voice/health */
export interface VoiceHealth {
  status: 'ok' | 'loading' | 'unavailable';
  reason?: string;
  missing?: string[];
  hint?: string;
}

interface SingleEnvelope<T> {
  success: boolean;
  result: T;
}

export const voice = {
  /**
   * GET /voice/health — public probe (no auth, always 200), so it opts out of
   * redirect-on-401: a failure should mark voice unavailable, never yank the
   * user to /login/. Envelope is { success, result: { status, ... } }.
   */
  async health(): Promise<VoiceHealth> {
    const body = await api.get<SingleEnvelope<VoiceHealth>>(
      '/api/voice/health',
      { redirectOnAuthError: false },
    );
    return body.result;
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
   * GET /voice/transcript/<id> — the speech already synthesized for one
   * settled reply. Returns the raw Response so the player can decode the body
   * itself: 200 carries the audio/wav, 202 means synthesis is running (this
   * request started it, for history that predates pre-synthesis), 409 means it
   * failed for good, 404 means the row can never speak.
   */
  transcript(transcriptId: number, signal?: AbortSignal): Promise<Response> {
    return fetch(`${getHost().replace(/\/$/, '')}/api/voice/transcript/${transcriptId}`, {
      method: 'GET',
      credentials: 'same-origin',
      signal,
    });
  },
};
