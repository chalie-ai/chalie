import { escHtml, showToast } from './utils.js';

/**
 * Document upload — dialog, drag-drop, processing, synthesis confirmation.
 */
export class DocumentUpload {
  constructor({ api, getHost }) {
    this._api = api;
    this._getHost = getHost;
  }

  init() {
    const dialog = document.getElementById('uploadDialog');
    const closeBtn = document.getElementById('uploadDialogClose');
    const dropzone = document.getElementById('uploadDropzone');
    const fileInput = document.getElementById('uploadFileInput');

    if (!dialog) return;

    closeBtn?.addEventListener('click', () => dialog.close());
    dialog.addEventListener('click', (e) => { if (e.target === dialog) dialog.close(); });

    // Dropzone click → file picker
    dropzone?.addEventListener('click', () => fileInput?.click());

    // Drag & drop
    dropzone?.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
    dropzone?.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
    dropzone?.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
      if (e.dataTransfer?.files?.length) this._handleFiles(e.dataTransfer.files);
    });

    // File input change
    fileInput?.addEventListener('change', () => {
      if (fileInput.files?.length) this._handleFiles(fileInput.files);
    });
  }

  openDialog() {
    const dialog = document.getElementById('uploadDialog');
    this._resetUploadDialog();
    dialog?.showModal();
  }

  /**
   * Upload a single file programmatically (e.g. from a drag-drop onto the viewport).
   * Opens the upload dialog to show progress, then handles the file.
   *
   * @param {File} file
   */
  uploadFile(file) {
    if (!file) return;
    this.openDialog();
    this._handleFiles([file]);
  }

  _resetUploadDialog() {
    const progress = document.getElementById('uploadProgress');
    const dupWarning = document.getElementById('uploadDuplicateWarning');
    const fileInput = document.getElementById('uploadFileInput');
    progress?.classList.add('hidden');
    dupWarning?.classList.add('hidden');
    if (fileInput) fileInput.value = '';
  }

  async _handleFiles(files) {
    const file = files[0];
    if (!file) return;

    const dialog = document.getElementById('uploadDialog');
    const progress = document.getElementById('uploadProgress');
    const progressLabel = document.getElementById('uploadProgressLabel');
    const dupWarning = document.getElementById('uploadDuplicateWarning');

    // Show progress
    progress?.classList.remove('hidden');
    progressLabel.textContent = 'Uploading...';
    dupWarning?.classList.add('hidden');

    try {
      const formData = new FormData();
      formData.append('file', file);

      const res = await this._api.upload('/documents/upload', formData);

      if (!res || res.error) {
        progressLabel.textContent = res?.error || 'Upload failed';
        progressLabel.style.color = '#f87171';
        return;
      }

      // Check for duplicates
      if (res.duplicates?.length) {
        const dup = res.duplicates[0];
        const dateStr = dup.created_at ? new Date(dup.created_at).toLocaleDateString() : '';
        dupWarning.innerHTML = `
          <div>This looks like an updated version of <strong>${escHtml(dup.original_name)}</strong>${dateStr ? ` from ${dateStr}` : ''}. Replace the older version, or keep both?</div>
          <div class="upload-duplicate__actions">
            <button class="upload-duplicate__btn upload-duplicate__btn--primary" data-action="replace" data-new-id="${res.id}" data-old-id="${dup.id}">Replace</button>
            <button class="upload-duplicate__btn" data-action="keep">Keep Both</button>
          </div>`;
        dupWarning.classList.remove('hidden');

        dupWarning.querySelectorAll('button').forEach(btn => {
          btn.addEventListener('click', () => {
            dupWarning.classList.add('hidden');
            if (btn.dataset.action === 'replace') {
              const newId = btn.dataset.newId;
              const oldId = btn.dataset.oldId;
              this._api.post(`/documents/${newId}/supersede`, { old_id: oldId })
                .then(() => showToast('Replaced older version'))
                .catch(() => showToast('Could not replace — keeping both'));
            }
          });
        });
      }

      // Poll for status
      progressLabel.textContent = 'Extracting text...';
      this._pollDocumentStatus(res.id, progressLabel, dialog);

    } catch (e) {
      progressLabel.textContent = 'Upload failed';
      progressLabel.style.color = '#f87171';
    }
  }

  async _pollDocumentStatus(docId, label, dialog) {
    let attempts = 0;
    const maxAttempts = 60; // 2 minutes max
    let synthWaitAttempts = 0;
    const maxSynthWait = 15; // 30s max to wait for LLM synthesis before showing card anyway

    const poll = async () => {
      if (attempts++ > maxAttempts) {
        label.textContent = 'Processing taking longer than expected...';
        return;
      }

      try {
        const res = await this._api.get(`/documents/${docId}`);
        const status = res?.item?.status;

        if (status === 'ready') {
          label.textContent = 'Ready';
          label.style.color = '#34d399';
          setTimeout(() => dialog?.close(), 1500);
          showToast(`Document "${res.item.original_name}" processed`);
          return;
        } else if (status === 'failed') {
          label.textContent = res?.item?.error_message || 'Processing failed';
          label.style.color = '#f87171';
          return;
        } else if (status === 'awaiting_confirmation') {
          const hasSynthesis = res.item.extracted_metadata?._synthesis;
          if (!hasSynthesis && synthWaitAttempts++ < maxSynthWait) {
            // Synthesis LLM call still in progress — wait up to 30s then proceed anyway
            label.textContent = 'Generating summary...';
            setTimeout(poll, 2000);
            return;
          }
          dialog?.close();
          this._showDocumentSynthesis(docId, res.item);
          return;
        } else if (status === 'processing') {
          label.textContent = 'Understanding document...';
        }
      } catch { /* retry */ }

      setTimeout(poll, 2000);
    };

    poll();
  }

  _showDocumentSynthesis(docId, doc) {
    const meta = doc.extracted_metadata || {};
    const synthesis = meta._synthesis || doc.summary || '';
    const keyFacts = meta._key_facts || [];
    const docType = (meta.document_type || {}).value || '';
    const name = escHtml(doc.original_name || 'Document');
    const typeTag = docType && docType !== 'document'
      ? `<span class="doc-synthesis__tag">${escHtml(docType)}</span>` : '';

    const factsHtml = keyFacts.length
      ? `<div class="doc-synthesis__facts">${keyFacts.map(f =>
          `<span class="doc-synthesis__fact">${escHtml(f)}</span>`).join('')}</div>`
      : '';

    const card = document.createElement('div');
    card.className = 'doc-synthesis-card';
    card.dataset.docId = docId;
    card.innerHTML = `
      <div class="doc-synthesis__header">
        <span class="doc-synthesis__name">${name}</span>
        ${typeTag}
      </div>
      <p class="doc-synthesis__text">${escHtml(synthesis)}</p>
      ${factsHtml}
      <div class="doc-synthesis__actions">
        <button class="doc-synthesis__btn doc-synthesis__btn--confirm">Looks good</button>
        <button class="doc-synthesis__btn doc-synthesis__btn--augment">Add context</button>
        <button class="doc-synthesis__btn doc-synthesis__btn--discard">Discard</button>
      </div>
      <div class="doc-synthesis__augment-area hidden">
        <textarea class="doc-synthesis__textarea"
          placeholder="Add context about this document..."></textarea>
        <button class="doc-synthesis__btn doc-synthesis__btn--submit">Save</button>
      </div>`;

    // Wire button handlers
    const confirmBtn = card.querySelector('.doc-synthesis__btn--confirm');
    const augmentBtn = card.querySelector('.doc-synthesis__btn--augment');
    const discardBtn = card.querySelector('.doc-synthesis__btn--discard');
    const augmentArea = card.querySelector('.doc-synthesis__augment-area');
    const submitBtn = card.querySelector('.doc-synthesis__btn--submit');
    const textarea = card.querySelector('.doc-synthesis__textarea');

    confirmBtn.addEventListener('click', async () => {
      confirmBtn.disabled = true;
      confirmBtn.textContent = 'Confirming...';
      try {
        await this._api.post(`/documents/${docId}/confirm`, {});
        card.remove();
        showToast('Document ready');
      } catch (e) {
        confirmBtn.disabled = false;
        confirmBtn.textContent = 'Looks good';
        showToast('Confirmation failed');
      }
    });

    augmentBtn.addEventListener('click', () => {
      augmentArea.classList.toggle('hidden');
      textarea.focus();
    });

    submitBtn.addEventListener('click', async () => {
      const ctx = textarea.value.trim();
      if (!ctx) return;
      submitBtn.disabled = true;
      submitBtn.textContent = 'Saving...';
      try {
        await this._api.post(`/documents/${docId}/augment`, { context: ctx });
        card.remove();
        showToast('Document ready with your context');
      } catch (e) {
        submitBtn.disabled = false;
        submitBtn.textContent = 'Save';
        showToast('Failed to save context');
      }
    });

    discardBtn.addEventListener('click', async () => {
      discardBtn.disabled = true;
      discardBtn.textContent = 'Discarding...';
      try {
        await this._api.del(`/documents/${docId}/purge`);
        card.remove();
        showToast('Document discarded');
      } catch (e) {
        discardBtn.disabled = false;
        discardBtn.textContent = 'Discard';
        showToast('Discard failed');
      }
    });

    // Inject into chat spine
    const spine = document.getElementById('conversationSpine');
    if (spine) spine.appendChild(card);
    card.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }
}
