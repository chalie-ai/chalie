/**
 * Wrappers API — mints an external bearer token for device pairing.
 *
 *   POST /api/wrappers/-1 → 201 envelope { success, result: { wrapper_id, token } }
 *     `token` is the raw bearer, shown ONCE and not recoverable. It becomes
 *     PairingPayload.token. `wrapper_id` is retained for later revocation via
 *     DELETE /api/wrappers/<wrapper_id>. Backed by
 *     backend/api/endpoints/wrappers.py (cookie-only POST — only a human
 *     cookie session can mint).
 */
import { api } from '@chalie/shared';

interface WrapperCreated {
  wrapper_id: string;
  token: string;
}

interface Envelope<T> {
  success: true;
  result: T;
}

interface EnvelopeError {
  success: false;
  result: never[];
  error: string;
}

type ApiResponse<T> = Envelope<T> | EnvelopeError;

export const wrappers = {
  /**
   * Mint a new wrapper token. Cookie-only endpoint; returns the raw token
   * exactly once (shown on creation, never on reads).
   */
  async create(input: { name: string }): Promise<WrapperCreated> {
    const res = (await api.post<ApiResponse<WrapperCreated>>('/api/wrappers/-1', input)) as ApiResponse<WrapperCreated>;
    if (!res.success) throw new Error(res.error);
    return res.result;
  },
};
