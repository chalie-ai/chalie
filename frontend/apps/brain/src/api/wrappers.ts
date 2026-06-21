/**
 * Wrappers API — mints an external bearer token for device pairing.
 *
 *   POST /api/wrappers → 201 { wrapper_id, token }
 *     `token` is the raw bearer, shown ONCE and not recoverable. It becomes
 *     PairingPayload.token. `wrapper_id` is retained for later revocation via
 *     DELETE /api/wrappers/<wrapper_id>. Backed by backend/api/wrappers.py
 *     (@require_auth + @_cookie_only — only a human cookie session can mint).
 */
import { api } from '@chalie/shared';

export const wrappers = {
  create(input: { name: string }): Promise<{ wrapper_id: string; token: string }> {
    return api.post('/api/wrappers', input);
  },
};
