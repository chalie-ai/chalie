import { api } from '@chalie/shared';

/** Payload sent to POST /api/policies/respond. */
export interface PolicyRespondPayload {
  request_id: string;
  approved: boolean;
}

export const policies = {
  /**
   * POST /api/policies/respond — resolve a pending permission gate, waking the
   * blocked ACT dispatch thread with the user's allow/deny decision.
   */
  respond(payload: PolicyRespondPayload): Promise<{ ok: boolean }> {
    return api.post('/api/policies/respond', payload);
  },
};
