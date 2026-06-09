import { showToast } from './utils.js';

/**
 * File attachment — hold raw File objects, preview strip, send with the message.
 *
 * No pre-upload round-trip: the raw files ride the multipart POST /chat at send
 * time and the backend ingests each via `document.upload` (by PATH, never bytes).
 * Images and documents both go through the same path:
 *   1. User picks or drops a file.
 *   2. A preview chip is rendered immediately (image thumbnail or doc icon) and
 *      the File is held in memory — nothing is sent to the server yet.
 *   3. On send, every held File is appended to the multipart /chat request as a
 *      `files` part (see ws.js `_postChat`).
 */
export class ImageAttach {
  /**
   * @param {{ getHost: () => string, onDocumentDrop?: (file: File) => void }} opts
   */
  constructor({ getHost, onDocumentDrop }) {
    this._getHost = getHost;
    this._onDocumentDrop = onDocumentDrop || null;

    // [{file: File, filename: string, element: HTMLElement, objectUrl, isImage}]
    this._attachments = [];
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
   * Process one file — hold it and show a preview chip.
   *
   * Nothing is sent to the server here; the raw File is held and uploaded as
   * part of the multipart POST /chat at send time. Images render as thumbnails;
   * other file types render as a doc icon with the filename.
   *
   * @param {File} file
   */
  handleFile(file) {
    if (this._attachments.length >= 10) {
      showToast('Maximum 10 attachments per message');
      return;
    }

    const isImage = file.type.startsWith('image/');
    // Object URL is created once and reused: the preview chip displays it now,
    // and the sent-message bubble reuses it so the image stays visible in the
    // user's own turn after send (the strip is cleared). Not revoked — the
    // browser releases it on unload, and a handful per session is negligible.
    const objectUrl = isImage ? URL.createObjectURL(file) : null;
    const chipEl = isImage
      ? this._addImageChip(file, objectUrl)
      : this._addDocChip(file.name);

    this._attachments.push({ file, filename: file.name, element: chipEl, objectUrl, isImage });
    this._updateSendBtn();
  }

  /**
   * Returns the raw File objects for all held attachments, to be appended to
   * the multipart POST /chat request.
   * @returns {File[]}
   */
  getFiles() {
    return this._attachments.map(a => a.file);
  }

  /**
   * Returns preview metadata for rendering attachments inside the sent-message
   * bubble. Must be read BEFORE clear() — clear() drops the attachment list.
   * @returns {Array<{filename: string, objectUrl: string|null, isImage: boolean}>}
   */
  getAttachments() {
    return this._attachments.map(a => ({
      filename: a.filename,
      objectUrl: a.objectUrl || null,
      isImage: !!a.isImage,
    }));
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
   * Number of currently attached files.
   * @returns {number}
   */
  get count() {
    return this._attachments.length;
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
   *
   * @param {File} file
   * @param {string} objectUrl — pre-created object URL for the file
   * @returns {HTMLElement}
   */
  _addImageChip(file, objectUrl) {
    const strip = document.getElementById('imagePreview');
    strip.classList.remove('hidden');

    const thumb = document.createElement('div');
    thumb.className = 'image-preview__thumb';

    const img = document.createElement('img');
    img.src = objectUrl || URL.createObjectURL(file);
    img.alt = file.name;
    thumb.appendChild(img);

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
    chip.className = 'image-preview__thumb image-preview__thumb--doc';

    const icon = document.createElement('div');
    icon.className = 'image-preview__doc-icon';
    icon.setAttribute('aria-hidden', 'true');
    icon.textContent = '📄';
    chip.appendChild(icon);

    const label = document.createElement('span');
    label.className = 'image-preview__doc-name';
    label.textContent = filename.length > 20 ? filename.slice(0, 18) + '…' : filename;
    chip.appendChild(label);

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
   * Enable the send button when there is text OR at least one attachment.
   */
  _updateSendBtn() {
    const sendBtn = document.getElementById('sendBtn');
    const textarea = document.getElementById('messageInput');
    if (!sendBtn) return;
    sendBtn.disabled = !((textarea?.value.trim()) || this._attachments.length > 0);
  }
}
