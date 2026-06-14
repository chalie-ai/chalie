import { getHost } from '@chalie/shared';
import { useApiClient } from '@chalie/shared';

/** A document duplicate detected at upload time. */
export interface DocumentDuplicate {
  id: string;
  original_name: string;
  match_type: string;
  created_at: string | null;
}

/** Response from POST /documents/upload (HTTP 201). */
export interface DocumentUploadResult {
  id: string;
  original_name: string;
  status: string;
  file_size: number;
  file_hash: string;
  duplicates?: DocumentDuplicate[];
}

/** A document item returned by GET /documents/<id>. */
export interface DocumentItem {
  id: string;
  original_name: string;
  status: 'pending' | 'processing' | 'ready' | 'failed' | 'awaiting_confirmation';
  [k: string]: unknown;
}

export const documents = {
  /**
   * POST /documents/upload — multipart upload.
   * Returns raw Response so the caller can inspect status / handle duplicates.
   */
  upload(file: File): Promise<Response> {
    const host = getHost();
    const base = host ? host.replace(/\/$/, '') : '';
    const formData = new FormData();
    formData.append('file', file, file.name);
    return fetch(`${base}/documents/upload`, {
      method: 'POST',
      credentials: 'same-origin',
      body: formData,
    });
  },

  /**
   * GET /documents/<id> — poll for processing status.
   * Legacy document_upload.js reads res?.item?.status.
   */
  status(id: string): Promise<{ item: DocumentItem }> {
    const api = useApiClient();
    return api.get(`/documents/${encodeURIComponent(id)}`);
  },
};
