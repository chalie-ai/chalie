import { showToast } from './utils.js';

/**
 * Image attachment — upload, preview strip, analysis tracking.
 */
export class ImageAttach {
  /**
   * @param {{ getHost: () => string, onDocumentDrop?: (file: File) => void }} opts
   */
  constructor({ getHost, onDocumentDrop }) {
    this._getHost = getHost;
    this._onDocumentDrop = onDocumentDrop || null;

    // [{id: string, element: HTMLElement}]
    this._attachedImages = [];

    // image_id → {element: HTMLElement, timeout: number}
    // Populated by handleFile; cleared by handleImageReady or timeout.
    this._pendingImageAnalysis = new Map();
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
   * Process one image file — upload, show preview, track analysis.
   *
   * Uploads the file via REST, then keeps the spinner and `analyzing` CSS class
   * on the thumbnail until the server pushes an `image_ready` WebSocket event.
   * A 90-second safety-net timeout replaces the spinner with a warning badge if
   * the event never arrives.
   *
   * The send button is enabled as soon as the server returns an `image_id` so
   * the user is not blocked.
   *
   * @param {File} file
   */
  async handleFile(file) {
    if (this._attachedImages.length >= 3) {
      showToast('Maximum 3 images per message');
      return;
    }
    const thumbEl = this._addImagePreview(file);

    const formData = new FormData();
    formData.append('image', file);
    try {
      const res = await fetch('/chat/image', {
        method: 'POST',
        credentials: 'same-origin',
        body: formData,
      });
      const data = await res.json();
      if (res.ok && data.image_id) {
        this._attachedImages.push({ id: data.image_id, element: thumbEl });
        // Keep the spinner and 'analyzing' class — they are cleared when the
        // server sends an 'image_ready' WebSocket event after background
        // analysis completes (see handleImageReady).  Do NOT remove them here;
        // the previous behaviour removed them immediately (before analysis even
        // started), which caused the LLM to silently miss image context.
        document.getElementById('sendBtn').disabled = false;

        // Safety net: if the 'image_ready' event does not arrive within 90 s,
        // replace the spinner with a warning badge so the user knows analysis
        // timed out.  The image remains attached.
        const timeoutId = setTimeout(() => {
          this._pendingImageAnalysis.delete(data.image_id);
          thumbEl.classList.remove('analyzing');
          thumbEl.querySelector('.image-preview__spinner')?.remove();
          const warn = document.createElement('span');
          warn.className = 'image-preview__warn';
          warn.title = 'Image analysis timed out — context may be unavailable';
          warn.textContent = '⚠';
          thumbEl.appendChild(warn);
        }, 90_000);

        this._pendingImageAnalysis.set(data.image_id, { element: thumbEl, timeout: timeoutId });
      } else {
        thumbEl.remove();
        this._updatePreviewVisibility();
        showToast(data.error || 'Image upload failed');
      }
    } catch {
      thumbEl.remove();
      this._updatePreviewVisibility();
      showToast('Image upload failed');
    }
  }

  /**
   * Called from the event router on WS `image_ready` events.
   * Removes the analyzing spinner on success, or shows an error badge on failure.
   *
   * @param {{ image_id: string, status: string }} data
   */
  handleImageReady(data) {
    const pending = this._pendingImageAnalysis.get(data.image_id);
    if (pending) {
      clearTimeout(pending.timeout);
      this._pendingImageAnalysis.delete(data.image_id);
      pending.element.classList.remove('analyzing');
      pending.element.querySelector('.image-preview__spinner')?.remove();
      if (data.status === 'failed') {
        // Surface the failure with an error badge on the thumbnail.
        const errBadge = document.createElement('span');
        errBadge.className = 'image-preview__error';
        errBadge.title = 'Image analysis failed — context unavailable';
        errBadge.textContent = '✕';
        pending.element.appendChild(errBadge);
        showToast('Image analysis failed');
      }
    }
  }

  /**
   * Returns array of attached image IDs (for message sending).
   * @returns {string[]}
   */
  getImageIds() {
    return this._attachedImages.map(a => a.id);
  }

  /**
   * Clears preview strip and cancels pending timeouts.
   *
   * Called when a message is sent or the user navigates away.  Cancels any
   * in-flight safety-net timers so they do not fire after the preview strip
   * has been cleared.
   */
  clear() {
    // Cancel all pending 90 s safety-net timers before clearing the DOM.
    for (const { timeout } of this._pendingImageAnalysis.values()) {
      clearTimeout(timeout);
    }
    this._pendingImageAnalysis.clear();

    this._attachedImages = [];
    const strip = document.getElementById('imagePreview');
    if (strip) {
      strip.innerHTML = '';
      strip.classList.add('hidden');
    }
  }

  /**
   * Number of currently attached images.
   * @returns {number}
   */
  get count() {
    return this._attachedImages.length;
  }

  /**
   * Wire drag-and-drop and paste listeners for image attachment.
   *
   * @param {{ dropzone: Element, pasteTarget: Element }} opts
   *   dropzone   — the element that receives dragover/drop events (e.g. .input-dock)
   *   pasteTarget — the element that receives paste events (e.g. #messageInput)
   */
  enableDropTargets({ dropzone, pasteTarget }) {
    if (!dropzone || !pasteTarget) return;

    // Dragover — prevent default to allow drop, add visual class
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

    // Dragleave — only remove class when leaving the dropzone entirely
    dropzone.addEventListener('dragleave', (ev) => {
      if (dropzone.contains(ev.relatedTarget)) return;
      dropzone.classList.remove('dragover');
    });

    // Drop on the input-dock
    dropzone.addEventListener('drop', (ev) => {
      ev.preventDefault(); // block browser navigation for dropped links/files
      dropzone.classList.remove('dragover');
      const files = ev.dataTransfer?.files;
      if (!files?.length) return;
      let dropped = 0;
      for (const file of files) {
        if (file.type.startsWith('image/')) {
          if (this._attachedImages.length + dropped >= 3) {
            showToast('Maximum 3 images per message');
            break;
          }
          this.handleFile(file);
          dropped++;
        } else if (this._onDocumentDrop) {
          this._onDocumentDrop(file);
        } else {
          showToast('Drop an image here (documents: use + menu)');
        }
      }
    });

    // Paste — only intercept when clipboard contains an image item
    pasteTarget.addEventListener('paste', (ev) => {
      const items = ev.clipboardData?.items;
      if (!items) return;
      for (const item of items) {
        if (item.type.startsWith('image/')) {
          const file = item.getAsFile();
          if (!file) continue; // browser returned null (e.g. Firefox async) — try next item
          ev.preventDefault();
          this.handleFile(file);
          return;
        }
      }
      // Plain text — fall through to native paste behaviour
    });
  }

  // ---------------------------------------------------------------------------
  // Private helpers
  // ---------------------------------------------------------------------------

  /**
   * Create and insert a thumbnail element into the preview strip.
   * The thumb starts with `analyzing` class and a spinner overlay.
   *
   * @param {File} file
   * @returns {HTMLElement} thumb element
   */
  _addImagePreview(file) {
    const strip = document.getElementById('imagePreview');
    strip.classList.remove('hidden');

    const thumb = document.createElement('div');
    thumb.className = 'image-preview__thumb analyzing';

    const img = document.createElement('img');
    img.src = URL.createObjectURL(file);
    img.alt = file.name;
    thumb.appendChild(img);

    // Spinner overlay (shown while upload/analysis in-flight)
    const spinner = document.createElement('div');
    spinner.className = 'image-preview__spinner';
    spinner.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v4m0 12v4m-7.07-3.93 2.83-2.83m8.48-8.48 2.83-2.83M2 12h4m12 0h4m-3.93 7.07-2.83-2.83M7.76 7.76 4.93 4.93"/></svg>';
    thumb.appendChild(spinner);

    // Remove button
    const removeBtn = document.createElement('button');
    removeBtn.className = 'image-preview__remove';
    removeBtn.setAttribute('aria-label', 'Remove image');
    removeBtn.textContent = '\u00d7';
    removeBtn.addEventListener('click', () => {
      const idx = this._attachedImages.findIndex(a => a.element === thumb);
      if (idx >= 0) this._attachedImages.splice(idx, 1);
      thumb.remove();
      this._updatePreviewVisibility();
      if (!this._attachedImages.length) {
        const textarea = document.getElementById('messageInput');
        document.getElementById('sendBtn').disabled = !textarea?.value.trim();
      }
    });
    thumb.appendChild(removeBtn);

    strip.appendChild(thumb);
    return thumb;
  }

  _updatePreviewVisibility() {
    const strip = document.getElementById('imagePreview');
    if (strip && !strip.children.length) strip.classList.add('hidden');
  }
}
