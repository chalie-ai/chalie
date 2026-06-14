import { getHost } from '@chalie/shared';

/**
 * Tips API — /api/tips/dismiss and /api/tips/mute.
 *
 * NOTE: These routes are called by legacy quick_tip_card.js but no corresponding
 * backend route was found in the backend/api/ directory. They are typed as opaque
 * and use raw fetch (matching the legacy implementation) so they fail gracefully
 * if the backend lacks the endpoint.
 */
export const tips = {
  /**
   * POST /api/tips/dismiss — dismiss a single tip by id.
   */
  dismiss(id: string): Promise<Response> {
    const host = getHost();
    const base = host ? host.replace(/\/$/, '') : '';
    return fetch(`${base}/api/tips/dismiss`, {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id }),
    });
  },

  /**
   * POST /api/tips/mute — mute an entire tip category.
   */
  mute(category: string): Promise<Response> {
    const host = getHost();
    const base = host ? host.replace(/\/$/, '') : '';
    return fetch(`${base}/api/tips/mute`, {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ category }),
    });
  },
};
