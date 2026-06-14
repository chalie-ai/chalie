/**
 * Brain API — endpoints derived from frontend/brain/brain.js.
 *
 * These are the ONLY /brain/-prefixed endpoints (confirmed in backend/api/brain.py):
 *   GET  /brain/info                → { db_size_human, document_count, ... }
 *   POST /brain/export              → binary octet-stream download (raw Response)
 *   POST /brain/import              → multipart import
 *   POST /brain/export/upload       → JSON export-to-cloud (github)
 *   POST /brain/github/test         → test github connection { github_token }
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

export interface BrainExportBody {
  include_files: boolean;
  encrypt: boolean;
  password: string | null;
}

export interface BrainUploadBody extends BrainExportBody {
  github_repo: string;
  github_token: string;
}

export interface BrainImportResult {
  db_size: number;
  files_restored: number;
  [key: string]: unknown;
}

export interface BrainUploadResult {
  commit_url?: string;
  [key: string]: unknown;
}

export interface GithubTestResult {
  username?: string;
  [key: string]: unknown;
}

export const brain = {
  info(): Promise<BrainInfo> {
    const api = useApiClient();
    return withAuth(() => api.get('/brain/info'));
  },

  /**
   * POST /brain/export → returns the raw Response so the caller can handle the blob.
   * Throws AuthError on 401 (via withAuth); does NOT throw on other non-ok statuses —
   * the caller inspects res.ok and reads .blob()/.json() itself.
   */
  export(body: BrainExportBody): Promise<Response> {
    const api = useApiClient();
    return withAuth(() => api.download('/brain/export', body));
  },

  import(formData: FormData): Promise<BrainImportResult> {
    const api = useApiClient();
    return withAuth(() => api.upload('/brain/import', formData));
  },

  exportUpload(body: BrainUploadBody): Promise<BrainUploadResult> {
    const api = useApiClient();
    return withAuth(() => api.post('/brain/export/upload', body));
  },

  testGithub(body: { github_token: string }): Promise<GithubTestResult> {
    const api = useApiClient();
    return withAuth(() => api.post('/brain/github/test', body));
  },
};
