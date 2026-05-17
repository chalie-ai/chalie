import { escHtml, lsGet, lsSet } from './utils.js';

/**
 * Update banner and dialog — version notifications and apply.
 */
export class UpdateSystem {
  constructor({ getHost }) {
    this._getHost = getHost;
    this._pendingUpdate = null;
  }

  init() {
    document.getElementById('updateBannerBtn')?.addEventListener('click', () => this._showUpdateDialog());
    document.getElementById('updateBannerDismiss')?.addEventListener('click', () => this._dismissUpdateBanner());
    document.getElementById('updateDialogClose')?.addEventListener('click', () => this._closeUpdateDialog());
    document.getElementById('updateCancelBtn')?.addEventListener('click', () => this._closeUpdateDialog());
    document.getElementById('updateApplyBtn')?.addEventListener('click', () => this._applyUpdate());
  }

  handleUpdateEvent(data) {
    this._pendingUpdate = data;
    const dismissedVersion = lsGet('chalie_update_dismissed');
    if (dismissedVersion === data.latest_tag) return;
    this._showUpdateBanner(data);
  }

  _showUpdateBanner(data) {
    const banner = document.getElementById('updateBanner');
    const versionEl = document.getElementById('updateBannerVersion');
    if (!banner || !versionEl) return;

    versionEl.textContent = `v${data.latest_version}`;

    if (data.deployment_mode === 'docker' || data.deployment_mode === 'dev') {
      const btn = document.getElementById('updateBannerBtn');
      if (btn) btn.textContent = 'Details';
    }

    banner.classList.remove('hidden');
  }

  _dismissUpdateBanner() {
    const banner = document.getElementById('updateBanner');
    if (banner) banner.classList.add('hidden');
    if (this._pendingUpdate) {
      lsSet('chalie_update_dismissed', this._pendingUpdate.latest_tag);
    }
  }

  _showUpdateDialog() {
    const dialog = document.getElementById('updateDialog');
    if (!dialog || !this._pendingUpdate) return;

    const data = this._pendingUpdate;
    this._populateDialogVersionInfo(data);
    this._applyDialogDeploymentMode(data);
    dialog.showModal();
  }

  /** Populate version labels and release notes in the dialog. */
  _populateDialogVersionInfo(data) {
    const currentEl = document.getElementById('updateCurrentVer');
    const newEl = document.getElementById('updateNewVer');
    const notesEl = document.getElementById('updateNotes');
    const actionsEl = document.getElementById('updateActions');
    const progressEl = document.getElementById('updateProgress');
    const instructionsEl = document.getElementById('updateInstructions');

    if (currentEl) currentEl.textContent = `v${data.current_version}`;
    if (newEl) newEl.textContent = `v${data.latest_version}`;
    if (notesEl) notesEl.textContent = data.release_notes || 'No release notes.';

    if (actionsEl) actionsEl.classList.remove('hidden');
    if (progressEl) progressEl.classList.add('hidden');
    if (instructionsEl) instructionsEl.classList.add('hidden');
  }

  /**
   * Swap the dialog body for deployment-mode-specific instructions when
   * the apply button cannot be offered (docker or dev modes).
   */
  _applyDialogDeploymentMode(data) {
    const actionsEl = document.getElementById('updateActions');
    const instructionsEl = document.getElementById('updateInstructions');

    if (data.deployment_mode === 'docker') {
      if (actionsEl) actionsEl.classList.add('hidden');
      if (instructionsEl) {
        instructionsEl.innerHTML = `<p>You're running Chalie in Docker. To update:</p><code>docker pull chalie/chalie:${escHtml(data.latest_tag)}\ndocker compose up -d</code>`;
        instructionsEl.classList.remove('hidden');
      }
    } else if (data.deployment_mode === 'dev') {
      if (actionsEl) actionsEl.classList.add('hidden');
      if (instructionsEl) {
        instructionsEl.innerHTML = `<p>You're running from a git clone. To update:</p><code>git pull origin main</code>`;
        instructionsEl.classList.remove('hidden');
      }
    }
  }

  _closeUpdateDialog() {
    document.getElementById('updateDialog')?.close();
  }

  async _applyUpdate() {
    if (!this._pendingUpdate) return;

    const actionsEl = document.getElementById('updateActions');
    const progressEl = document.getElementById('updateProgress');
    if (actionsEl) actionsEl.classList.add('hidden');
    if (progressEl) progressEl.classList.remove('hidden');

    try {
      const resp = await fetch('/system/update/apply', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify({ tag: this._pendingUpdate.latest_tag }),
      });
      const result = await resp.json();
      this._handleApplyResult(result, actionsEl, progressEl);
    } catch {
      const statusEl = progressEl?.querySelector('.update-dialog__status');
      if (statusEl) statusEl.textContent = 'Update request failed.';
    }
  }

  /**
   * Update the progress UI based on the /system/update/apply response.
   * On failure, restores the action buttons after a short delay.
   */
  _handleApplyResult(result, actionsEl, progressEl) {
    const statusEl = progressEl?.querySelector('.update-dialog__status');
    if (!result.ok) {
      if (statusEl) statusEl.textContent = result.message || 'Update failed.';
      setTimeout(() => {
        if (actionsEl) actionsEl.classList.remove('hidden');
        if (progressEl) progressEl.classList.add('hidden');
      }, 3000);
      return;
    }
    if (statusEl) statusEl.textContent = 'Restarting Chalie...';
  }
}
