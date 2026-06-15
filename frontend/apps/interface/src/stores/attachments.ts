/**
 * Attachments store — holds raw File objects (images AND documents) and renders
 * a preview chip for each. Nothing is sent to the server here.
 *
 * Port source: frontend/interface/image_attach.js
 *
 * No pre-upload round-trip: the raw files ride the multipart POST /chat at send
 * time and the backend ingests each via `document.upload` (by PATH, never bytes)
 * — see session.sendMessage. Images and documents both go through the same path:
 *   1. User picks, drops, or pastes a file.
 *   2. A preview chip is rendered immediately (image thumbnail or doc icon) and
 *      the File is held in memory — nothing is sent to the server yet.
 *   3. On send, every held File is appended to the multipart /chat request.
 *
 * Image previews are generated via webPlatformAdapter.readFileAsDataURL
 * (PlatformAdapter contract — no URL.createObjectURL). Documents have no
 * thumbnail (dataUrl null) and render as a 📄 chip.
 */
import { defineStore } from 'pinia';
import { webPlatformAdapter } from '@chalie/shared';
import { showToast } from '../utils/toast';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** Preview metadata for a single attachment chip in the strip. */
export interface AttachmentPreview {
  filename: string;
  /** data: URL produced by FileReader — safe to use as <img src>. Null for docs. */
  dataUrl: string | null;
  isImage: boolean;
}

/** Internal entry — the raw File plus its preview chip (parallel to legacy _attachments). */
interface AttachmentEntry {
  file: File;
  preview: AttachmentPreview;
}

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

export const useAttachmentsStore = defineStore('attachments', {
  state: () => ({
    /** Raw file entries — cap 10 per legacy image_attach.js:54. */
    _entries: [] as AttachmentEntry[],

    /** Reactive preview list for the strip component. */
    previews: [] as AttachmentPreview[],
  }),

  actions: {
    /**
     * Add files from any source (file-input change, drag-drop, paste).
     *
     * Images and documents are both held for send and given a preview chip —
     * the only difference is the chip (thumbnail vs 📄 icon).
     */
    async addFiles(files: FileList | File[]): Promise<void> {
      for (const file of Array.from(files)) {
        await this._addFile(file);
      }
    },

    /** Remove an attachment by index (× button in the strip). */
    remove(index: number): void {
      if (index < 0 || index >= this._entries.length) return;
      this._entries.splice(index, 1);
      this.previews.splice(index, 1);
    },

    /**
     * Return raw File objects for all held attachments, appended to the
     * multipart POST /chat request. Must be read BEFORE clear().
     */
    getFiles(): File[] {
      return this._entries.map((e) => e.file);
    },

    /**
     * Clear all previews and held attachments.
     * Called by the controller after a message is sent.
     */
    clear(): void {
      this._entries = [];
      this.previews = [];
    },

    // ── Internal helpers ───────────────────────────────────────────────────

    async _addFile(file: File): Promise<void> {
      if (this._entries.length >= 10) {
        // Cap matched from legacy image_attach.js:54.
        showToast('Maximum 10 attachments per message');
        return;
      }
      const isImage = file.type.startsWith('image/');
      let dataUrl: string | null = null;
      if (isImage) {
        try {
          dataUrl = await webPlatformAdapter.readFileAsDataURL(file);
        } catch {
          // Preview fails gracefully — chip still shown without thumbnail.
        }
      }
      const preview: AttachmentPreview = { filename: file.name, dataUrl, isImage };
      this._entries.push({ file, preview });
      this.previews.push(preview);
    },
  },
});
