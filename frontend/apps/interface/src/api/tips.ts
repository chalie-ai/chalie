import { getHost } from '@chalie/shared';

/**
 * Tips API — /api/tips/dismiss and /api/tips/mute.
 *
 * NOTE: These routes are dormant (no backend emitter currently exists for
 * quick_tip events). They are ported faithfully so they activate correctly
 * if/when the backend ships the endpoint.
 *
 * Body shapes match the legacy quick_tip_card.js exactly:
 *   dismiss → { tip_id: id }       (legacy line 129: JSON.stringify({ tip_id: tipId }))
 *   mute    → {}                   (legacy line 142-146: no body, credentials only)
 */
export const tips = {
  /**
   * POST /api/tips/dismiss — mark a single tip as seen by its tip_id.
   */
  dismiss(id: string): Promise<Response> {
    return fetch(`${getHost().replace(/\/$/, '')}/api/tips/dismiss`, {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tip_id: id }),
    });
  },

  /**
   * POST /api/tips/mute — disable all tips globally.
   * Legacy quick_tip_card.js lines 141-146: POST with Content-Type header, no body.
   */
  mute(): Promise<Response> {
    return fetch(`${getHost().replace(/\/$/, '')}/api/tips/mute`, {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
    });
  },
};
