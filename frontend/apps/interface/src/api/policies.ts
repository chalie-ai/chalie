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
   *
   * Answers 204 with an empty body on success — including when the gate is
   * unknown or already resolved, which is a graceful no-op by design.
   */
  async respond(payload: PolicyRespondPayload): Promise<void> {
    await api.post('/api/policies/respond', payload);
  },
};
