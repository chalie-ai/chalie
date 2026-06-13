const PanelBrain = (() => {
  let _root = null;
  let _info = null;

  function mount(root, sub) {
    _root = root;
    _render();
    _loadInfo();
  }

  function unmount() {
    _root = null;
  }

  function _render() {
    if (!_root) return;
    _root.innerHTML = `
      <div class="panel-header">
        <h2>${Icons.Brain(20)} Brain</h2>
      </div>
      <div id="brainContent">
        <div class="loading">Loading brain info…</div>
      </div>`;
  }

  async function _loadInfo() {
    try {
      const res = await BrainApp.apiFetch("/brain/info");
      if (!res.ok) throw new Error("Failed");
      _info = await res.json();
      _renderContent();
    } catch (e) {
      const el = document.getElementById("brainContent");
      if (el) el.innerHTML = `<div class="empty-state"><p>Could not load brain info.</p></div>`;
    }
  }

  function _renderContent() {
    const el = document.getElementById("brainContent");
    if (!el || !_info) return;
    const esc = BrainApp.escapeHtml;
    const dbSize = esc(_info.db_size_human || "0 B");
    const docCount = _info.document_count || 0;

    el.innerHTML = `
      <div class="brain-overview">
        <div class="brain-stats">
          <div class="export-card">
            <div class="export-card-icon">${Icons.Database(24)}</div>
            <div class="export-card-label">Database</div>
            <div class="export-card-value">${dbSize}</div>
          </div>
          <div class="export-card">
            <div class="export-card-icon">${Icons.Document(24)}</div>
            <div class="export-card-label">Documents</div>
            <div class="export-card-value">${docCount}</div>
          </div>
        </div>

        <div class="brain-actions">
          <button class="btn btn-primary" id="brainExportBtn">${Icons.Download(14)} Export Brain</button>
          <button class="btn btn-secondary" id="brainImportBtn">${Icons.Upload(14)} Restore from Backup</button>
        </div>

        <div class="brain-section">
          <h3 class="brain-section-title">GitHub Backup</h3>
          <div class="brain-form-grid">
            <label class="form-label">Repository
              <input type="text" class="form-input" id="ghRepo" placeholder="owner/chalie-brain-backup">
            </label>
            <label class="form-label">Token
              <input type="password" class="form-input" id="ghToken" placeholder="ghp_...">
            </label>
          </div>
          <div class="brain-form-actions">
            <button class="btn btn-sm btn-secondary" id="ghTestBtn">Test Connection</button>
            <button class="btn btn-sm btn-primary" id="ghUploadBtn">Upload Backup</button>
            <span class="form-hint" id="ghStatus"></span>
          </div>
        </div>
      </div>`;

    document.getElementById("brainExportBtn").addEventListener("click", _showExportDialog);
    document.getElementById("brainImportBtn").addEventListener("click", _showImportDialog);
    document.getElementById("ghTestBtn").addEventListener("click", _testGitHub);
    document.getElementById("ghUploadBtn").addEventListener("click", _showGitHubUploadDialog);
  }

  function _showExportDialog() {
    _showModal({
      title: "Export Brain",
      body: `
        <label class="form-label"><input type="checkbox" id="expFiles" checked> Include document files</label>
        <label class="form-label"><input type="checkbox" id="expEncrypt" checked> Encrypt with passphrase</label>
        <label class="form-label">Passphrase
          <input type="password" class="form-input" id="expPass" autocomplete="new-password">
        </label>
        <label class="form-label">Confirm passphrase
          <input type="password" class="form-input" id="expPassConfirm" autocomplete="new-password">
        </label>`,
      confirmLabel: "Export",
      confirmClass: "btn btn-primary",
      async onConfirm() {
        await _doExport();
      },
    });
  }

  async function _doExport() {
    const includeFiles = document.getElementById("expFiles")?.checked ?? true;
    const encrypt = document.getElementById("expEncrypt")?.checked ?? true;
    let password = document.getElementById("expPass")?.value || "";

    if (encrypt) {
      const confirm = document.getElementById("expPassConfirm")?.value || "";
      if (!password || password.length < 4) {
        BrainApp.showToast("Passphrase must be at least 4 characters", "error");
        return;
      }
      if (password !== confirm) {
        BrainApp.showToast("Passphrases do not match", "error");
        return;
      }
    }

    BrainApp.showToast("Exporting brain…", "info");
    try {
      const res = await BrainApp.apiFetch("/brain/export", {
        method: "POST",
        body: JSON.stringify({ include_files: includeFiles, encrypt, password: encrypt ? password : null }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ error: "Export failed" }));
        throw new Error(err.error);
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `brain-export-${new Date().toISOString().replace(/[:.]/g, "-")}.chalie-backup`;
      a.click();
      URL.revokeObjectURL(url);
      BrainApp.showToast("Brain exported successfully", "success");
    } catch (e) {
      BrainApp.showToast(e.message || "Export failed", "error");
    }
  }

  function _showImportDialog() {
    _showConfirm({
      title: "Restore from Backup",
      desc: "This will <strong>replace all current data</strong>. A pre-restore snapshot is saved automatically.",
      confirmLabel: "Choose Backup File",
      confirmClass: "btn btn-danger",
      onConfirm() {
        const input = document.createElement("input");
        input.type = "file";
        input.accept = ".chalie-backup";
        input.addEventListener("change", () => {
          if (input.files[0]) _promptPasswordAndImport(input.files[0]);
        });
        input.click();
      },
    });
  }

  function _promptPasswordAndImport(file) {
    _showModal({
      title: "Enter Passphrase",
      body: `
        <p>Decrypting: <strong>${BrainApp.escapeHtml(file.name)}</strong></p>
        <label class="form-label">Passphrase
          <input type="password" class="form-input" id="impPass" autocomplete="current-password">
        </label>`,
      confirmLabel: "Restore",
      confirmClass: "btn btn-danger",
      async onConfirm() {
        await _doImport(file);
      },
    });
  }

  async function _doImport(file) {
    const password = document.getElementById("impPass")?.value || "";
    if (!password) {
      BrainApp.showToast("Passphrase is required", "error");
      return;
    }

    BrainApp.showToast("Restoring brain…", "info");
    const form = new FormData();
    form.append("file", file);
    form.append("password", password);

    try {
      const res = await BrainApp.apiFetch("/brain/import", {
        method: "POST",
        body: form,
      }, true);
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Import failed");

      BrainApp.showToast(
        `Restored! DB: ${(data.db_size / 1024 / 1024).toFixed(1)} MB, Files: ${data.files_restored}`,
        "success",
        { duration: 6000 }
      );
      setTimeout(() => location.reload(), 2000);
    } catch (e) {
      BrainApp.showToast(e.message || "Import failed", "error");
    }
  }

  async function _testGitHub() {
    const token = document.getElementById("ghToken")?.value;
    const status = document.getElementById("ghStatus");
    if (!token) { BrainApp.showToast("Enter a GitHub token first", "error"); return; }
    if (status) status.textContent = "Testing…";
    try {
      const res = await BrainApp.apiFetch("/brain/github/test", {
        method: "POST",
        body: JSON.stringify({ github_token: token }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error);
      if (status) status.textContent = `Connected as @${data.username}`;
      BrainApp.showToast(`GitHub connected as @${data.username}`, "success");
    } catch (e) {
      if (status) status.textContent = "Connection failed";
      BrainApp.showToast(e.message || "GitHub test failed", "error");
    }
  }

  function _showGitHubUploadDialog() {
    _showModal({
      title: "Upload Backup to GitHub",
      body: `
        <p class="modal-desc">The backup will be encrypted and stored as a file in the repo.</p>
        <label class="form-label">Encryption passphrase (optional)
          <input type="password" class="form-input" id="ghUploadPass" autocomplete="new-password">
        </label>`,
      confirmLabel: "Upload",
      confirmClass: "btn btn-primary",
      async onConfirm() {
        await _doGitHubUpload();
      },
    });
  }

  async function _doGitHubUpload() {
    const repo = document.getElementById("ghRepo")?.value?.trim();
    const token = document.getElementById("ghToken")?.value?.trim();
    if (!repo || !token) { BrainApp.showToast("Enter repo and token", "error"); return; }

    const password = document.getElementById("ghUploadPass")?.value || "";
    const encrypt = password.length >= 4;
    if (password && password.length < 4) {
      BrainApp.showToast("Passphrase must be at least 4 characters", "error");
      return;
    }

    BrainApp.showToast("Exporting and uploading to GitHub…", "info");
    try {
      const res = await BrainApp.apiFetch("/brain/export/upload", {
        method: "POST",
        body: JSON.stringify({
          include_files: true,
          encrypt,
          password: encrypt ? password : null,
          github_repo: repo,
          github_token: token,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error);
      BrainApp.showToast("Backup uploaded to GitHub!", "success", { duration: 5000 });
      const status = document.getElementById("ghStatus");
      if (status && data.commit_url) {
        const safeUrl = data.commit_url.replace(/[^a-zA-Z0-9:\/_.\-]/g, "");
        status.textContent = "Last upload: " + new Date().toISOString().slice(0, 10);
      }
    } catch (e) {
      BrainApp.showToast(e.message || "Upload failed", "error");
    }
  }

  function _showModal({ title, body, confirmLabel, confirmClass, onConfirm }) {
    const overlay = document.createElement("div");
    overlay.className = "modal-overlay";
    const esc = BrainApp.escapeHtml;
    overlay.innerHTML = `<div class="modal">
      <div class="modal-header"><h3>${esc(title)}</h3></div>
      <div class="modal-body">${body}</div>
      <div class="modal-actions">
        <button class="btn btn-secondary" data-cancel>Cancel</button>
        <button class="${confirmClass}" data-confirm>${esc(confirmLabel)}</button>
      </div>
    </div>`;
    document.body.appendChild(overlay);
    overlay.querySelector("[data-cancel]").addEventListener("click", () => overlay.remove());
    overlay.addEventListener("click", (e) => { if (e.target === overlay) overlay.remove(); });
    overlay.querySelector("[data-confirm]").addEventListener("click", () => { overlay.remove(); onConfirm(); });
  }

  function _showConfirm({ title, desc, confirmLabel, confirmClass, onConfirm }) {
    const overlay = document.createElement("div");
    overlay.className = "modal-overlay";
    const esc = BrainApp.escapeHtml;
    overlay.innerHTML = `<div class="modal modal-sm">
      <div class="modal-header"><h3>${esc(title)}</h3></div>
      <p class="modal-desc">${desc}</p>
      <div class="modal-actions">
        <button class="btn btn-secondary" data-cancel>Cancel</button>
        <button class="${confirmClass}" data-confirm>${esc(confirmLabel)}</button>
      </div>
    </div>`;
    document.body.appendChild(overlay);
    overlay.querySelector("[data-cancel]").addEventListener("click", () => overlay.remove());
    overlay.addEventListener("click", (e) => { if (e.target === overlay) overlay.remove(); });
    overlay.querySelector("[data-confirm]").addEventListener("click", () => { overlay.remove(); onConfirm(); });
  }

  return { mount, unmount };
})();
