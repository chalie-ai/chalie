/**
 * Documents API — the Endpoint/Action contract under /api/documents.
 *
 * GET    /api/documents/all                    → listing envelope, page-walked → Document[]
 * GET    /api/documents/all?include_deleted=true → include deleted docs
 * GET    /api/watched-folders/all              → WatchedFolder[] (listing envelope, page-walked)
 * GET    /api/documents/:id                    → { success, result: Document } → unwrapped
 * DELETE /api/documents/:id                    → 204 soft delete
 * POST   /api/documents/upload                 → multipart upload → { success, result }
 */
import { api } from '@chalie/shared';
import { fetchAllPages } from './paginate';

export interface Document {
  id: string | number;
  title?: string | null;
  filename?: string | null;
  status?: string | null;
  deleted?: boolean;
  [key: string]: unknown;
}

// Serialized from the WatchedFolder response DTO of GET /api/watched-folders/all
// (the Endpoint-contract listing route). `last_scan_at`/`last_scan_files` are not
// part of that DTO — the panel renders them defensively and they read as empty.
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
  list(includeDeleted = false): Promise<Document[]> {
    return fetchAllPages<Document>(
      '/api/documents/all',
      includeDeleted ? { include_deleted: 'true' } : undefined,
    );
  },

  watchedFolders(): Promise<WatchedFolder[]> {
    return fetchAllPages<WatchedFolder>('/api/watched-folders/all');
  },

  get(id: string | number): Promise<Document> {
    return api.get<{ result: Document }>(`/api/documents/${id}`).then((r) => r.result);
  },

  delete(id: string | number): Promise<unknown> {
    return api.del(`/api/documents/${id}`);
  },

  upload(formData: FormData): Promise<unknown> {
    return api.upload('/api/documents/upload', formData);
  },
};
