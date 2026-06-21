/*
 * Import / Export panel — whole-instance snapshot Time-Machine.
 *
 * Export: optional password → POST /api/snapshot/export → download the .zip.
 * Import: file + optional password → confirm a FULL WIPE → multipart POST to
 * /api/snapshot/import, which stages the restore and restarts the instance.
 *
 * Reuses the established BrainApp panel idioms (apiFetch / showToast / blob
 * download) and the shared CSS classes (.export-card / .btn-* / .modal /
 * .form-input), all of which consume theme.css variables so the panel renders
 * correctly in both dark and light themes. Paired with api/snapshot.py.
 */
const PanelImportExport = (() => {
  let _root = null;

  function mount(root) {
    _root = root;
    _render();
  }

  function unmount() {
    _root = null;
  }

  function _render() {
    if (!_root) return;
    _root.innerHTML = `
      <div class="panel-header">
        <h2>${Icons.Backup(20)} Import / Export</h2>
      </div>
      <div class="brain-overview">
        <div class="export-card">
          <div class="export-card-icon">${Icons.Download(24)}</div>
          <div class="export-card-label">Export a full snapshot</div>
          <p class="form-hint">
            Downloads a single .zip that is a complete clone of this instance —
            databases, documents, skills, and secrets. Set a password to encrypt
            it with AES-256.
          </p>
          <label class="form-label">Password (optional)
            <input type="password" class="form-input" id="snapExportPass" autocomplete="new-password">
          </label>
          <button class="btn btn-primary" id="snapExportBtn">${Icons.Download(14)} Export Snapshot</button>
        </div>

        <div class="export-card">
          <div class="export-card-icon">${Icons.Upload(24)}</div>
          <div class="export-card-label">Restore from a snapshot</div>
          <p class="form-hint">
            Importing a snapshot <strong>completely wipes and overrides ALL
            existing data</strong> in this instance. This is a full restore, not
            a merge. The instance restarts to apply it.
          </p>
          <label class="form-label">Snapshot file
            <input type="file" class="form-input" id="snapImportFile" accept=".zip">
          </label>
          <label class="form-label">Password (if encrypted)
            <input type="password" class="form-input" id="snapImportPass" autocomplete="current-password">
          </label>
          <button class="btn btn-danger" id="snapImportBtn">${Icons.Upload(14)} Import &amp; Restore</button>
        </div>
      </div>`;

    document.getElementById("snapExportBtn").addEventListener("click", _doExport);
    document.getElementById("snapImportBtn").addEventListener("click", _confirmImport);
  }

  async function _doExport() {
    const password = document.getElementById("snapExportPass")?.value || "";
    const card =
      document.getElementById("snapExportBtn")?.closest(".export-card") ||
      _root?.querySelector(".export-card");
    _setCardLoading(card, true, "Preparing your export…");
    try {
      const res = await BrainApp.apiFetch("/api/snapshot/export", {
        method: "POST",
        body: JSON.stringify({ password: password || null }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ error: "Export failed" }));
        throw new Error(err.error || "Export failed");
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `chalie-snapshot-${new Date().toISOString().replace(/[:.]/g, "-")}.zip`;
      a.click();
      URL.revokeObjectURL(url);
      BrainApp.showToast("Snapshot exported", "success");
    } catch (e) {
      BrainApp.showToast(e.message || "Export failed", "error");
    } finally {
      _setCardLoading(card, false);
    }
  }

  /* While a long operation runs on an export-card, hide all its input fields
     and the action button, and show an inline loader with `message` in their
     place; restore them when called with `on=false`. Shared by export (assemble
     snapshot) and import (stage restore + restart).

     Toggle via inline display (not the `hidden` attribute): `.btn` and
     `.form-label` carry author `display` rules that override the low-priority
     `[hidden]` UA style, so the attribute alone wouldn't hide them. */
  function _setCardLoading(card, on, message) {
    if (!card) return;
    const fields = card.querySelectorAll(".form-label");
    const btn = card.querySelector("button");
    let loader = card.querySelector(".export-loader");

    if (on) {
      fields.forEach((f) => (f.style.display = "none"));
      if (btn) btn.style.display = "none";
      if (!loader) {
        loader = document.createElement("div");
        loader.className = "export-loader";
        loader.innerHTML =
          `<span class="export-spinner" aria-hidden="true"></span><span>${message}</span>`;
        card.appendChild(loader);
      }
      loader.style.display = "";
    } else {
      if (loader) loader.remove();
      fields.forEach((f) => (f.style.display = ""));
      if (btn) btn.style.display = "";
    }
  }

  function _confirmImport() {
    const fileInput = document.getElementById("snapImportFile");
    const file = fileInput?.files?.[0];
    if (!file) {
      BrainApp.showToast("Choose a snapshot file first", "error");
      return;
    }
    _showConfirm({
      title: "Restore from snapshot?",
      desc: `This will <strong>completely wipe and override ALL existing data</strong> with the contents of <strong>${BrainApp.escapeHtml(file.name)}</strong>. This is a full restore, not a merge, and cannot be undone. The instance will restart to apply it.`,
      confirmLabel: "Wipe & Restore",
      confirmClass: "btn btn-danger",
      onConfirm() { _doImport(file); },
    });
  }

  async function _doImport(file) {
    const password = document.getElementById("snapImportPass")?.value || "";
    const card = document.getElementById("snapImportBtn")?.closest(".export-card");
    const form = new FormData();
    form.append("file", file);
    if (password) form.append("password", password);

    _setCardLoading(card, true, "Restoring snapshot — the instance will restart…");
    BrainApp.showToast("Staging restore…", "info");
    try {
      const res = await BrainApp.apiFetch("/api/snapshot/import", { method: "POST", body: form }, true);
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.error || "Import failed");
      BrainApp.showToast("Restore staged — restarting to apply…", "success", { duration: 8000 });
      // Keep the loader up through the restart; the page reloads itself.
      setTimeout(() => location.reload(), 6000);
    } catch (e) {
      _setCardLoading(card, false);
      BrainApp.showToast(e.message || "Import failed", "error");
    }
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
