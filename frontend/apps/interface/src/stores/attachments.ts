/**
 * Attachments store — holds raw File objects (images AND documents) and renders
 * a preview chip for each. Nothing is sent to the server here; raw files ride
 * the multipart POST /api/thread at send time (backend ingests via `document.upload`
 * by PATH, never bytes — see session.sendMessage). Image previews come from
 * webPlatformAdapter.readFileAsDataURL (no URL.createObjectURL); docs have no
 * thumbnail (dataUrl null) and render as a 📄 chip.
 */
import { defineStore } from 'pinia';
import { webPlatformAdapter } from '@chalie/shared';
import { showToast } from '../utils/toast';

export interface AttachmentPreview {
  filename: string;
  /** data: URL produced by FileReader — safe as <img src>. Null for docs and
   *  for images whose read failed (chip still renders, just without a thumb). */
  dataUrl: string | null;
  isImage: boolean;
}

interface AttachmentEntry {
  file: File;
  preview: AttachmentPreview;
}

export const useAttachmentsStore = defineStore('attachments', {
  state: () => ({
    _entries: [] as AttachmentEntry[],
    previews: [] as AttachmentPreview[],
  }),

  actions: {
    async addFiles(files: FileList | File[]): Promise<void> {
      for (const file of Array.from(files)) {
        await this._addFile(file);
      }
    },

    remove(index: number): void {
      if (index < 0 || index >= this._entries.length) return;
      this._entries.splice(index, 1);
      this.previews.splice(index, 1);
    },

    /** Raw Files for all held attachments. Must be read BEFORE clear(). */
    getFiles(): File[] {
      return this._entries.map((e) => e.file);
    },

    clear(): void {
      this._entries = [];
      this.previews = [];
    },

    async _addFile(file: File): Promise<void> {
      if (this._entries.length >= 10) {
        showToast('Maximum 10 attachments per message');
        return;
      }
      const isImage = file.type.startsWith('image/');
      // Read the data: URL BEFORE pushing so the preview is complete the moment
      // it enters the reactive array. Pushing a raw object and then mutating it
      // (preview.dataUrl = …) does NOT re-render — Vue tracks the reactive proxy
      // the template reads, not the raw ref we'd mutate, so the thumbnail would
      // land only on the next unrelated re-render. A FileReader read of a
      // just-picked local file is near-instant, so there is no perceptible gap.
      let dataUrl: string | null = null;
      if (isImage) {
        try {
          dataUrl = await webPlatformAdapter.readFileAsDataURL(file);
        } catch {
          // Preview fails gracefully — chip still shown without a thumbnail.
        }
      }
      const preview: AttachmentPreview = { filename: file.name, dataUrl, isImage };
      this._entries.push({ file, preview });
      this.previews.push(preview);
    },
  },
});
