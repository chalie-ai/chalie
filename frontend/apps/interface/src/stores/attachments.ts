/**
 * Attachments store — owns image-attachment state AND document upload/confirm flow.
 *
 * Port sources:
 *   - frontend/interface/image_attach.js  (preview strip, 10-file cap)
 *   - frontend/interface/document_upload.js (upload, poll, synthesis card)
 *
 * Image files are held as raw File objects; previews are generated via
 * webPlatformAdapter.readFileAsDataURL (PlatformAdapter contract — no
 * URL.createObjectURL). Non-image files are routed directly into the document
 * upload flow.
 *
 * Document flow:
 *   uploadDocument → POST /upload → duplicate? → expose duplicate state
 *   → poll every 2 s (max 60 attempts) → awaiting_confirmation → expose docCard
 *   → user confirms / augments / discards → REST → clear docCard
 */
import { defineStore } from 'pinia';
import { webPlatformAdapter } from '@chalie/shared';
import { documents } from '../api/documents';
import type { DocumentDuplicate } from '../api/documents';
import { showToast } from '../utils/toast';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** Preview metadata for a single attachment chip in the strip. */
export interface AttachmentPreview {
  filename: string;
  /** data: URL produced by FileReader — safe to use as <img src>. */
  dataUrl: string | null;
  isImage: boolean;
}

/** A detected duplicate exposed while the user chooses Replace / Keep Both. */
export interface DuplicateState {
  newId: string;
  dup: DocumentDuplicate;
}

/** Synthesis card data exposed once the document reaches awaiting_confirmation. */
export interface DocCard {
  id: string;
  name: string;
  synthesis: string;
  keyFacts: string[];
  typeLabel: string;
}

/** Internal image entry (parallel to legacy _attachments array). */
interface ImageEntry {
  file: File;
  preview: AttachmentPreview;
}

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

export const useAttachmentsStore = defineStore('attachments', {
  state: () => ({
    /** Raw image entries — cap 10 per legacy image_attach.js:51. */
    _images: [] as ImageEntry[],

    /** Reactive preview list for the strip component. */
    previews: [] as AttachmentPreview[],

    /** Duplicate state while waiting for user Replace/Keep choice. */
    duplicate: null as DuplicateState | null,

    /** Synthesis card — non-null once a document reaches awaiting_confirmation. */
    docCard: null as DocCard | null,

    /** Error from the most recent document upload/poll cycle. */
    docError: null as string | null,
  }),

  actions: {
    // ── Image/file entry point ─────────────────────────────────────────────

    /**
     * Add files from any source (file-input change, drag-drop, paste).
     *
     * Images → hold for send + generate preview chip.
     * Non-images → immediately kick off uploadDocument.
     */
    async addFiles(files: FileList | File[]): Promise<void> {
      const list = Array.from(files);
      for (const file of list) {
        if (file.type.startsWith('image/')) {
          await this._addImage(file);
        } else {
          void this.uploadDocument(file);
        }
      }
    },

    /** Remove an image attachment by index (× button in the strip). */
    removeImage(index: number): void {
      if (index < 0 || index >= this._images.length) return;
      this._images.splice(index, 1);
      this.previews.splice(index, 1);
    },

    /**
     * Return raw File objects for all held image attachments.
     * Called by the controller's InputDock before send.
     * Must be read BEFORE clear().
     */
    getFiles(): File[] {
      return this._images.map((e) => e.file);
    },

    /**
     * Clear all image previews and attachments.
     * Called by the controller after a message is sent.
     */
    clear(): void {
      this._images = [];
      this.previews = [];
    },

    // ── Document upload flow ───────────────────────────────────────────────

    /**
     * Upload a non-image file:
     *   1. Add a doc chip preview immediately.
     *   2. POST to /documents/upload.
     *   3. If duplicates present, expose duplicate state and stop (user decides).
     *   4. Otherwise start polling.
     */
    async uploadDocument(file: File): Promise<void> {
      this.docError = null;

      // Show a doc-chip in the strip immediately (mirrors legacy _addDocChip)
      const preview: AttachmentPreview = {
        filename: file.name,
        dataUrl: null,
        isImage: false,
      };
      this.previews.push(preview);
      const previewIndex = this.previews.length - 1;

      let uploadResult: Awaited<ReturnType<typeof documents.upload>>;
      try {
        uploadResult = await documents.upload(file);
      } catch (err) {
        console.warn('[attachments] upload failed:', err);
        this.docError = 'Upload failed';
        showToast('Upload failed');
        this.previews.splice(previewIndex, 1);
        return;
      }

      if (uploadResult.duplicates?.length) {
        // Expose duplicate state — user must choose Replace or Keep Both.
        this.duplicate = { newId: uploadResult.id, dup: uploadResult.duplicates[0] };
        // Remove the optimistic chip; the card UI owns the interaction now.
        this.previews.splice(previewIndex, 1);
        return;
      }

      // Remove the optimistic chip; polling will show the synthesis card on completion.
      this.previews.splice(previewIndex, 1);
      await this._pollDocumentStatus(uploadResult.id);
    },

    /**
     * Replace an older duplicate with the newly uploaded document.
     * Calls POST /documents/<newId>/supersede, then polls the new document.
     */
    async replaceDuplicate(newId: string, oldId: string): Promise<void> {
      this.duplicate = null;
      try {
        await documents.supersede(newId, oldId);
        showToast('Replaced older version');
      } catch {
        // Best-effort — if supersede fails, still poll so the doc is usable.
        showToast('Could not replace — keeping both');
      }
      await this._pollDocumentStatus(newId);
    },

    /** User chose "Keep Both" — just poll the new document as-is. */
    async keepBothDuplicate(newId: string): Promise<void> {
      this.duplicate = null;
      await this._pollDocumentStatus(newId);
    },

    // ── Synthesis card actions (called from DocumentConfirmCard) ───────────

    async confirmDoc(id: string): Promise<void> {
      await documents.confirm(id);
      this.docCard = null;
    },

    async augmentDoc(id: string, context: string): Promise<void> {
      await documents.augment(id, context);
      this.docCard = null;
    },

    async discardDoc(id: string): Promise<void> {
      await documents.discard(id);
      this.docCard = null;
    },

    // ── Internal helpers ───────────────────────────────────────────────────

    async _addImage(file: File): Promise<void> {
      if (this._images.length >= 10) {
        // Cap matched from legacy image_attach.js:51-54.
        showToast('Maximum 10 attachments per message');
        return;
      }
      let dataUrl: string | null = null;
      try {
        dataUrl = await webPlatformAdapter.readFileAsDataURL(file);
      } catch {
        // Preview fails gracefully — chip still shown without thumbnail.
      }
      const preview: AttachmentPreview = {
        filename: file.name,
        dataUrl,
        isImage: true,
      };
      this._images.push({ file, preview });
      this.previews.push(preview);
    },

    /**
     * Poll GET /documents/<id> every 2000 ms, up to 60 attempts.
     *
     * Matches legacy _pollDocumentStatus logic exactly:
     *   - 'ready'                → auto-clear (no card needed).
     *   - 'failed'               → expose docError.
     *   - 'awaiting_confirmation'→ wait up to 15 extra polls (30 s) for
     *                              _synthesis to appear, then show the card anyway.
     *   - anything else          → keep polling.
     */
    async _pollDocumentStatus(docId: string): Promise<void> {
      const MAX_ATTEMPTS = 60;
      const MAX_SYNTH_WAIT = 15; // extra polls waiting for LLM synthesis (matches legacy)
      const INTERVAL_MS = 2000;

      let attempts = 0;
      let synthWaitAttempts = 0;

      const poll = async (): Promise<void> => {
        if (attempts++ > MAX_ATTEMPTS) {
          this.docError = 'Processing taking longer than expected...';
          showToast('Processing taking longer than expected...');
          return;
        }

        try {
          const res = await documents.status(docId);
          const item = res?.item;
          const status = item?.status;

          if (status === 'ready') {
            // Document is ready — no confirmation needed (legacy document_upload.js:151).
            showToast(`Document "${item.original_name}" processed`);
            return;
          }

          if (status === 'failed') {
            // `||` (not `??`) to match legacy document_upload.js:154 — also falls
            // back when error_message is an empty string.
            this.docError = item.error_message || 'Processing failed';
            showToast(this.docError);
            return;
          }

          if (status === 'awaiting_confirmation') {
            const hasSynthesis = !!item.extracted_metadata?._synthesis;
            if (!hasSynthesis && synthWaitAttempts++ < MAX_SYNTH_WAIT) {
              // LLM synthesis still in progress — wait up to 30 s then show card anyway.
              await this._delay(INTERVAL_MS);
              return poll();
            }
            // Expose the synthesis card.
            const meta = item.extracted_metadata ?? {};
            this.docCard = {
              id: docId,
              name: item.original_name,
              synthesis: meta._synthesis ?? '',
              keyFacts: meta._key_facts ?? [],
              typeLabel:
                typeof meta.document_type === 'object' && meta.document_type !== null
                  ? (meta.document_type as { value?: string }).value ?? ''
                  : '',
            };
            return;
          }

          // Still pending/processing — keep polling.
        } catch (err) {
          console.warn('[attachments] status poll failed:', err);
        }

        await this._delay(INTERVAL_MS);
        return poll();
      };

      await poll();
    },

    _delay(ms: number): Promise<void> {
      return new Promise((resolve) => setTimeout(resolve, ms));
    },
  },
});
