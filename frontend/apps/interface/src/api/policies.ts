import { api } from '@chalie/shared';

/** Payload sent to POST /api/policies/respond. */
export interface PolicyRespondPayload {
  request_id: string;
  approved: boolean;
}

/**
 * The interactive turn a permission gate belongs to — the backend stamps it
 * on every `permission_request` frame and pending-gate listing so the
 * interface can route the card to the lane that turn renders in.
 */
export interface PermissionOrigin {
  /** ConfigType of the turn: `user` or `scheduled` (a gate never fires for a turn with no surface). */
  type: string;
  /** The turn the gated tool call runs under (a schedule's item id on `scheduled`). */
  turn_id: number;
  /** True for a reply sent into a thread (the panel's turn), false for a main-spine turn. */
  forked: boolean;
}

/**
 * One pending permission gate — the SAME fields the `permission_request` WS
 * frame carries (minus `type`): the backend lists the frames it broadcast, so
 * a card restored over REST is the card the socket pushed.
 */
export interface PendingPermission {
  /** Opaque ID; resolves the gate on /api/policies/respond. */
  request_id: string;
  /** Permission key the policy gated, e.g. "pim" or "email.send". */
  action_id: string;
  /** The model's one-line summary of the gated action — the card's title. */
  summary: string;
  origin: PermissionOrigin;
}

interface ListingEnvelope<T> {
  success: true;
  result: T[];
  pagination: { page: number; limit: number; total: number };
}

export const policies = {
  /**
   * GET /api/policies/pending — every permission gate still waiting for an
   * answer, oldest first. Envelope is { success, result: [...], pagination }
   * (the same listing shape as GET /api/policies/blocked). Fetched on every
   * WS connect so a reload or a dropped socket brings the cards back: the
   * backend thread keeps waiting regardless, the frame was only the trigger.
   */
  async pending(): Promise<PendingPermission[]> {
    const body = await api.get<ListingEnvelope<PendingPermission>>('/api/policies/pending');
    return body.result;
  },

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
