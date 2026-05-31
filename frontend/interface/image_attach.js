import { showToast } from './utils.js';

/**
 * File attachment — upload via POST /upload, preview strip, send as attachments.
 *
 * Replaces the old image-only flow (POST /chat/image + image_ids + base64 files).
 * Images and documents both go through the same path:
 *   1. User picks or drops a file.
 *   2. POST /upload — saves to /tmp on the server, returns {tmp_path, filename,
 *      content_type, size}.
 *   3. A preview chip is rendered in the strip (image thumbnail or doc icon).
 *   4. The send button is blocked while any upload is in progress.
 *   5. On send, all tmp_paths are passed as the `attachments` array.
 */
export class ImageAttach {
  /**
   * @param {{ getHost: () => string, onDocumentDrop?: (file: File) => void }} opts
   */
  constructor({ getHost, onDocumentDrop }) {
    this._getHost = getHost;
    this._onDocumentDrop = onDocumentDrop || null;

    // [{tmpPath: string, filename: string, element: HTMLElement}]
    this._attachments = [];

    // Number of uploads currently in-flight — send is blocked while > 0.
    this._uploadsInProgress = 0;
  }

  // ---------------------------------------------------------------------------
  // Public API
  // ---------------------------------------------------------------------------

  /**
   * Bind the file input change handler.  Must be called once the DOM is ready.
   */
  init() {
    document.getElementById('imageFileInput')?.addEventListener('change', (e) => {
      if (e.target.files?.length) this.handleFile(e.target.files[0]);
      e.target.value = '';
    });
  }

  /**
   * Process one file — upload to /upload, show preview chip.
   *
   * The send button is disabled for the duration of the upload and re-evaluated
   * once the upload completes (success or failure).  Images render as thumbnails;
   * other file types render as a doc icon with the filename.
   *
   * @param {File} file
   */
  async handleFile(file) {
    if (this._attachments.length >= 10) {
      showToast('Maximum 10 attachments per message');
      return;
    }

    const isImage = file.type.startsWith('image/');
    const chipEl = isImage
      ? this._addImageChip(file)
      : this._addDocChip(file.name);

    this._uploadsInProgress++;
    this._updateSendBtn();

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/upload', {
        method: 'POST',
        credentials: 'same-origin',
        body: formData,
      });
      const data = await res.json();

      if (res.ok && data.tmp_path) {
        // Remove the in-progress spinner and mark as ready.
        chipEl.classList.remove('analyzing');
        chipEl.querySelector('.image-preview__spinner')?.remove();
        this._attachments.push({ tmpPath: data.tmp_path, filename: data.filename, element: chipEl });
      } else {
        chipEl.remove();
        this._updatePreviewVisibility();
        showToast(data.error || 'Upload failed');
      }
    } catch {
      chipEl.remove();
      this._updatePreviewVisibility();
      showToast('Upload failed');
    } finally {
      this._uploadsInProgress--;
      this._updateSendBtn();
    }
  }

  /**
   * Returns the list of tmp_paths for all successfully uploaded attachments.
   * @returns {string[]}
   */
  getAttachmentPaths() {
    return this._attachments.map(a => a.tmpPath);
  }

  /**
   * Clears the preview strip.
   *
   * Called when a message is sent or the user navigates away.
   */
  clear() {
    this._attachments = [];
    const strip = document.getElementById('imagePreview');
    if (strip) {
      strip.innerHTML = '';
      strip.classList.add('hidden');
    }
  }

  /**
   * Number of currently attached files (uploaded or in-flight).
   * @returns {number}
   */
  get count() {
    return this._attachments.length + this._uploadsInProgress;
  }

  /**
   * True while at least one upload is in progress.
   * @returns {boolean}
   */
  get isUploading() {
    return this._uploadsInProgress > 0;
  }

  /**
   * Wire drag-and-drop and paste listeners for file attachment.
   *
   * @param {{ dropzone: Element, pasteTarget: Element }} opts
   */
  enableDropTargets({ dropzone, pasteTarget }) {
    if (!dropzone || !pasteTarget) return;

    dropzone.addEventListener('dragenter', (ev) => {
      if (!ev.dataTransfer?.types?.includes('Files')) return;
      ev.preventDefault();
      dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragover', (ev) => {
      if (!ev.dataTransfer?.types?.includes('Files')) return;
      ev.preventDefault();
      dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', (ev) => {
      if (dropzone.contains(ev.relatedTarget)) return;
      dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (ev) => {
      ev.preventDefault();
      dropzone.classList.remove('dragover');
      const files = ev.dataTransfer?.files;
      if (!files?.length) return;
      for (const file of files) {
        if (file.type.startsWith('image/')) {
          this.handleFile(file);
        } else if (this._onDocumentDrop) {
          this._onDocumentDrop(file);
        } else {
          showToast('Drop an image or document here');
        }
      }
    });

    pasteTarget.addEventListener('paste', (ev) => {
      const items = ev.clipboardData?.items;
      if (!items) return;
      for (const item of items) {
        if (item.type.startsWith('image/')) {
          const file = item.getAsFile();
          if (!file) continue;
          ev.preventDefault();
          this.handleFile(file);
          return;
        }
      }
    });
  }

  // ---------------------------------------------------------------------------
  // Private helpers
  // ---------------------------------------------------------------------------

  /**
   * Add an image thumbnail chip to the preview strip.
   * The chip starts with the `analyzing` class and a spinner while uploading.
   *
   * @param {File} file
   * @returns {HTMLElement}
   */
  _addImageChip(file) {
    const strip = document.getElementById('imagePreview');
    strip.classList.remove('hidden');

    const thumb = document.createElement('div');
    thumb.className = 'image-preview__thumb analyzing';

    const img = document.createElement('img');
    img.src = URL.createObjectURL(file);
    img.alt = file.name;
    thumb.appendChild(img);

    const spinner = document.createElement('div');
    spinner.className = 'image-preview__spinner';
    spinner.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4m0 12v4m-7.07-3.93 2.83-2.83m8.48-8.48 2.83-2.83M2 12h4m12 0h4m-3.93 7.07-2.83-2.83M7.76 7.76 4.93 4.93"/></svg>';
    thumb.appendChild(spinner);

    thumb.appendChild(this._makeRemoveBtn(thumb));
    strip.appendChild(thumb);
    return thumb;
  }

  /**
   * Add a document chip (icon + filename) to the preview strip.
   *
   * @param {string} filename
   * @returns {HTMLElement}
   */
  _addDocChip(filename) {
    const strip = document.getElementById('imagePreview');
    strip.classList.remove('hidden');

    const chip = document.createElement('div');
    chip.className = 'image-preview__thumb image-preview__thumb--doc analyzing';

    const icon = document.createElement('div');
    icon.className = 'image-preview__doc-icon';
    icon.setAttribute('aria-hidden', 'true');
    icon.textContent = '📄';
    chip.appendChild(icon);

    const label = document.createElement('span');
    label.className = 'image-preview__doc-name';
    label.textContent = filename.length > 20 ? filename.slice(0, 18) + '…' : filename;
    chip.appendChild(label);

    const spinner = document.createElement('div');
    spinner.className = 'image-preview__spinner';
    spinner.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4m0 12v4m-7.07-3.93 2.83-2.83m8.48-8.48 2.83-2.83M2 12h4m12 0h4m-3.93 7.07-2.83-2.83M7.76 7.76 4.93 4.93"/></svg>';
    chip.appendChild(spinner);

    chip.appendChild(this._makeRemoveBtn(chip));
    strip.appendChild(chip);
    return chip;
  }

  /**
   * Build a × remove button that removes the chip and its attachment entry.
   *
   * @param {HTMLElement} chipEl
   * @returns {HTMLButtonElement}
   */
  _makeRemoveBtn(chipEl) {
    const btn = document.createElement('button');
    btn.className = 'image-preview__remove';
    btn.setAttribute('aria-label', 'Remove attachment');
    btn.textContent = '×';
    btn.addEventListener('click', () => {
      const idx = this._attachments.findIndex(a => a.element === chipEl);
      if (idx >= 0) this._attachments.splice(idx, 1);
      chipEl.remove();
      this._updatePreviewVisibility();
      this._updateSendBtn();
    });
    return btn;
  }

  _updatePreviewVisibility() {
    const strip = document.getElementById('imagePreview');
    if (strip && !strip.children.length) strip.classList.add('hidden');
  }

  /**
   * Enable the send button when there is text OR ready attachments AND no
   * upload is in progress.  Disabled while any upload is in-flight.
   */
  _updateSendBtn() {
    const sendBtn = document.getElementById('sendBtn');
    const textarea = document.getElementById('messageInput');
    if (!sendBtn) return;
    const hasContent = !!(textarea?.value.trim()) || this._attachments.length > 0;
    sendBtn.disabled = !hasContent || this._uploadsInProgress > 0;
  }
}
