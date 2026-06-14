/**
 * Brain API — endpoints derived from frontend/brain/brain.js.
 *
 * These are the ONLY /brain/-prefixed endpoints (confirmed in backend/api/brain.py):
 *   GET  /brain/info                → { db_size_human, document_count, ... }
 *   POST /brain/export              → binary octet-stream download
 *   POST /brain/import              → multipart import
 *   POST /brain/export/upload       → multipart export-to-cloud
 *   POST /brain/github/test         → test github connection { token, repo, ... }
 *
 * NOTE: timestamps are rendered AS-IS from the backend (spec requirement).
 * No date formatting is performed here.
 */
import { useApiClient } from '@chalie/shared';
import { withAuth } from './http';

export interface BrainInfo {
  db_size_human?: string | null;
  document_count?: number | null;
  [key: string]: unknown;
}

export const brain = {
  info(): Promise<BrainInfo> {
    const api = useApiClient();
    return withAuth(() => api.get('/brain/info'));
  },

  /**
   * POST /brain/export → triggers a binary download.
   * Returns the Response so the caller can handle the blob.
   */
  async export(): Promise<Response> {
    const res = await fetch('/brain/export', {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
    });
    if (res.status === 401) {
      window.location.replace(
        '/login/?next=' + encodeURIComponent(window.location.pathname),
      );
      throw new Error('redirecting to login');
    }
    return res;
  },

  import(formData: FormData): Promise<unknown> {
    const api = useApiClient();
    return withAuth(() => api.upload('/brain/import', formData));
  },

  exportUpload(formData: FormData): Promise<unknown> {
    const api = useApiClient();
    return withAuth(() => api.upload('/brain/export/upload', formData));
  },

  testGithub(body: Record<string, unknown>): Promise<{ success: boolean; error?: string }> {
    const api = useApiClient();
    return withAuth(() => api.post('/brain/github/test', body));
  },
};
