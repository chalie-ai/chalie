/**
 * Documents API — endpoints derived from frontend/brain/documents.js fetch calls.
 *
 * GET    /documents                       → { items: Document[] }
 * GET    /documents?include_deleted=true  → include deleted docs
 * GET    /api/watched-folders/all         → WatchedFolder[] (listing envelope, page-walked)
 * GET    /documents/:id                   → { item: Document }
 * DELETE /documents/:id                   → delete document
 * POST   /documents/upload                → multipart upload
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
    return api.get(includeDeleted ? '/api/documents?include_deleted=true' : '/api/documents');
  },

  watchedFolders(): Promise<WatchedFolder[]> {
    return fetchAllPages<WatchedFolder>('/api/watched-folders/all');
  },

  get(id: string | number): Promise<{ item: Document }> {
    return api.get(`/api/documents/${id}`);
  },

  delete(id: string | number): Promise<unknown> {
    return api.del(`/api/documents/${id}`);
  },

  upload(formData: FormData): Promise<unknown> {
    return api.upload('/api/documents/upload', formData);
  },
};
