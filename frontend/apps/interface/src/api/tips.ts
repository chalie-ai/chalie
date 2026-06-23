import { getHost } from '@chalie/shared';

/**
 * Tips API — /api/tips/dismiss and /api/tips/mute. These routes are dormant
 * (no backend emitter for quick_tip events yet); kept so they work if/when the
 * backend ships the endpoint.
 */
export const tips = {
  /** POST /api/tips/dismiss — mark a single tip seen by its tip_id. */
  dismiss(id: string): Promise<Response> {
    return fetch(`${getHost().replace(/\/$/, '')}/api/tips/dismiss`, {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tip_id: id }),
    });
  },

  /** POST /api/tips/mute — disable all tips globally (no body). */
  mute(): Promise<Response> {
    return fetch(`${getHost().replace(/\/$/, '')}/api/tips/mute`, {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
    });
  },
};
