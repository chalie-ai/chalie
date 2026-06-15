import { api } from '@chalie/shared';

/** A single attachment served from /documents/<id>/preview. */
export interface ConversationAttachment {
  doc_id: string;
  filename: string;
  mime_type: string;
  is_image: boolean;
  url: string;
}

/** A rich-media or plain-text content segment inside an assistant turn. */
export interface ConversationSegment {
  type: 'text' | 'rich';
  content?: string;
  tag?: string;
  payload?: Record<string, unknown>;
  synthesis?: string;
}

/** A single turn returned by /conversation/recent. */
export interface ConversationMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  /** Present on user turns when attachments were uploaded. */
  attachments?: ConversationAttachment[];
  /** Present on assistant turns — one or more content segments. */
  segments?: ConversationSegment[];
}

export const conversation = {
  /**
   * Fetch recent conversation turns.
   * @param limit  Max turns to return (1–120, default 12).
   * @param before Transcript ID offset — for history pagination (maps to `offset` on
   *               the legacy backend; the plan uses `before` as the param name).
   */
  recent(
    limit = 12,
    before?: number,
  ): Promise<{ messages: ConversationMessage[]; has_more: boolean }> {
    const q = new URLSearchParams({ limit: String(limit) });
    if (before != null) q.set('before', String(before));
    return api.get(`/conversation/recent?${q.toString()}`);
  },
};
