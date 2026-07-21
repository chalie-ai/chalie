/**
 * Snapshot API — whole-instance Import/Export Time-Machine.
 * Paired with the Action-contract routes in backend/api/actions/snapshot/*.
 * JSON responses use the { success, result, error? } envelope; export's
 * success body stays a raw binary stream.
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
  restarting: boolean;
}

interface SingleEnvelope<T> {
  success: true;
  result: T;
}

export const snapshot = {
  /**
   * Throws AuthError on 401; does NOT throw on other non-ok statuses — the
   * caller inspects res.ok and reads .blob()/.json() itself.
   */
  export(password: string | null): Promise<Response> {
    return api.download('/api/snapshot/export', { password });
  },

  async import(formData: FormData): Promise<SnapshotImportResult> {
    const res = await api.upload<SingleEnvelope<SnapshotImportResult>>(
      '/api/snapshot/import',
      formData,
    );
    return res.result;
  },
};
