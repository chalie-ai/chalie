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
  error_message?: string;
  extracted_metadata?: {
    _synthesis?: string;
    _key_facts?: string[];
    document_type?: { value?: string };
    [k: string]: unknown;
  };
  [k: string]: unknown;
}

export const documents = {
  /**
   * POST /documents/upload — multipart upload.
   * The raw fetch is used (no ApiClient) to keep getHost() parity with the
   * existing upload() implementation and avoid Content-Type conflicts.
   */
  upload(file: File): Promise<DocumentUploadResult> {
    const host = getHost();
    const base = host ? host.replace(/\/$/, '') : '';
    const formData = new FormData();
    formData.append('file', file, file.name);
    return fetch(`${base}/documents/upload`, {
      method: 'POST',
      credentials: 'same-origin',
      body: formData,
    }).then((res) => res.json() as Promise<DocumentUploadResult>);
  },

  /**
   * GET /documents/<id> — poll for processing status.
   * Legacy document_upload.js reads res?.item?.status.
   */
  status(id: string): Promise<{ item: DocumentItem }> {
    const api = useApiClient();
    return api.get(`/documents/${encodeURIComponent(id)}`);
  },

  /**
   * POST /documents/<id>/confirm — confirm the extracted synthesis.
   */
  confirm(id: string): Promise<{ ok: boolean; status: string }> {
    const api = useApiClient();
    return api.post(`/documents/${encodeURIComponent(id)}/confirm`);
  },

  /**
   * POST /documents/<id>/augment — add user context to the document.
   */
  augment(id: string, context: string): Promise<{ ok: boolean; status: string }> {
    const api = useApiClient();
    return api.post(`/documents/${encodeURIComponent(id)}/augment`, { context });
  },

  /**
   * DELETE /documents/<id>/purge — discard the document.
   */
  discard(id: string): Promise<{ ok: boolean }> {
    const api = useApiClient();
    return api.del(`/documents/${encodeURIComponent(id)}/purge`);
  },

  /**
   * POST /documents/<id>/supersede — replace an older duplicate document.
   */
  supersede(id: string, oldId: string): Promise<{ ok: boolean; supersedes_id: string }> {
    const api = useApiClient();
    return api.post(`/documents/${encodeURIComponent(id)}/supersede`, { old_id: oldId });
  },
};
