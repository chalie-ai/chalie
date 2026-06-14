/**
 * Documents API — endpoints derived from frontend/brain/documents.js fetch calls.
 *
 * GET    /documents                       → { items: Document[] }
 * GET    /documents?include_deleted=true  → include deleted docs
 * GET    /documents/watched-folders       → { items: WatchedFolder[] }
 * GET    /documents/:id                   → { item: Document }
 * DELETE /documents/:id                   → delete document
 * POST   /documents/upload                → multipart upload
 */
import { useApiClient } from '@chalie/shared';
import { withAuth } from './http';

export interface Document {
  id: string | number;
  title?: string | null;
  filename?: string | null;
  status?: string | null;
  deleted?: boolean;
  [key: string]: unknown;
}

// Fields mirror the raw `watched_folders` columns — GET /documents/watched-folders
// returns `dict(zip(cols, row))` from `FolderWatcherService.get_all_folders()`
// (no serialization layer), so the keys are the DB column names verbatim.
export interface WatchedFolder {
  id: string | number;
  folder_path: string;
  label?: string | null;
  enabled?: number | boolean;
  last_scan_at?: string | null;
  last_scan_files?: number;
  [key: string]: unknown;
}

export const documents = {
  list(includeDeleted = false): Promise<{ items: Document[] }> {
    const api = useApiClient();
    const url = includeDeleted ? '/documents?include_deleted=true' : '/documents';
    return withAuth(() => api.get(url));
  },

  watchedFolders(): Promise<{ items: WatchedFolder[] }> {
    const api = useApiClient();
    return withAuth(() => api.get('/documents/watched-folders'));
  },

  get(id: string | number): Promise<{ item: Document }> {
    const api = useApiClient();
    return withAuth(() => api.get(`/documents/${id}`));
  },

  delete(id: string | number): Promise<unknown> {
    const api = useApiClient();
    return withAuth(() => api.del(`/documents/${id}`));
  },

  upload(formData: FormData): Promise<unknown> {
    const api = useApiClient();
    return withAuth(() => api.upload('/documents/upload', formData));
  },
};
