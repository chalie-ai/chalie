/**
 * Snapshot API — whole-instance Import/Export Time-Machine (TKT-949).
 * Derived from frontend/brain/import_export.js; paired with backend/api/snapshot.py.
 *
 *   POST /api/snapshot/export → streams a single .zip clone of the instance
 *                               (databases, documents, skills, secrets).
 *                               Body { password } → AES-256 zip when set.
 *   POST /api/snapshot/import → multipart { file, password? }; stages a FULL
 *                               wipe-and-replace restore and restarts the
 *                               instance to apply it at next boot.
 */
import { api } from '@chalie/shared';

export interface SnapshotImportResult {
  ok?: boolean;
  restarting?: boolean;
  error?: string;
  [key: string]: unknown;
}

export const snapshot = {
  /**
   * POST /api/snapshot/export → raw Response so the caller handles the blob.
   * Throws AuthError on 401; does NOT throw on other non-ok statuses — the
   * caller inspects res.ok and reads .blob()/.json() itself.
   */
  export(password: string | null): Promise<Response> {
    return api.download('/api/snapshot/export', { password });
  },

  import(formData: FormData): Promise<SnapshotImportResult> {
    return api.upload('/api/snapshot/import', formData);
  },
};
