// Documents panel — document management + watched folders.
const PanelDocuments = (() => {
  let _root = null;
  let _sub = 'active';
  let _docs = [];
  let _folders = [];
  let _loaded = false;
  let _search = '';
  let _groupBy = 'all';

  function mount(root, sub) {
    _root = root;
    _sub = sub || 'active';
    _render();
  }

  function unmount() { _root = null; }

  function _render() {
    if (!_root) return;
    _root.innerHTML = `<div class="panel-header">
      <h2>Documents</h2>
      <div class="panel-header-actions">
        <input type="text" id="docSearchInput" class="search-input" placeholder="Search documents…" value="${BrainApp.escapeHtml(_search)}">
        <button class="btn btn-primary" id="docUploadBtn">${Icons.Upload(14)} Upload</button>
      </div>
    </div>
    <div class="doc-group-tabs" id="docGroupTabs">
      ${['all', 'doc_category', 'doc_project', 'doc_date'].map(g => {
        const label = { all: 'All', doc_category: 'Category', doc_project: 'Project', doc_date: 'Date' }[g];
        return `<button class="group-tab${g === _groupBy ? ' active' : ''}" data-group="${g}">${label}</button>`;
      }).join('')}
    </div>
    <div id="docsContent"><div class="loading">Loading…</div></div>`;

    document.getElementById('docSearchInput').addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { _search = e.target.value; _loaded = false; _load(); }
    });
    document.getElementById('docUploadBtn').addEventListener('click', _uploadDoc);
    document.getElementById('docGroupTabs').addEventListener('click', (e) => {
      const btn = e.target.closest('[data-group]');
      if (!btn) return;
      _groupBy = btn.dataset.group;
      _renderDocs();
      document.querySelectorAll('#docGroupTabs .group-tab').forEach(b => b.classList.toggle('active', b === btn));
    });

    if (!_loaded) _load();
    else _renderDocs();
  }

  async function _load() {
    try {
      const docUrl = _sub === 'deleted' ? '/documents?include_deleted=true' : '/documents';
      const [docRes, folderRes] = await Promise.all([
        BrainApp.apiFetch(docUrl),
        BrainApp.apiFetch('/documents/watched-folders'),
      ]);
      if (docRes.ok) { const d = await docRes.json(); _docs = d.items || []; }
      if (folderRes.ok) { const d = await folderRes.json(); _folders = d.items || []; }
      _loaded = true;
    } catch { BrainApp.showToast('Failed to load documents', 'error'); }
    _renderDocs();
  }

  function _renderDocs() {
    const el = document.getElementById('docsContent');
    if (!el) return;

    let filtered = _docs;
    if (_sub === 'active') filtered = _docs.filter(d => d.status === 'ready');
    else if (_sub === 'processing') filtered = _docs.filter(d => d.status === 'building' || d.status === 'processing' || d.status === 'pending');
    else if (_sub === 'uploads') filtered = _docs.filter(d => d.source_type === 'upload');
    else if (_sub === 'deleted') filtered = _docs.filter(d => d.deleted_at !== null);

    if (_search) {
      const q = _search.toLowerCase();
      filtered = filtered.filter(d => (d.original_name || '').toLowerCase().includes(q) || (d.doc_category || '').toLowerCase().includes(q));
    }

    let html = '';

    if (_folders.length > 0) {
      html += `<div class="watched-folders-section">
        <h4 class="section-head">Watched Folders</h4>
        <div class="watched-folders-list">${_folders.map(f => `<div class="watched-folder-item">
          <span class="wf-icon">${Icons.Eye(14)}</span>
          <span class="wf-label">${BrainApp.escapeHtml(f.label || f.path || '')}</span>
          <span class="wf-path">${BrainApp.escapeHtml(f.path || '')}</span>
          <span class="wf-meta">${f.files || 0} files · ${BrainApp.escapeHtml(f.last_scan || '')}</span>
          <span class="wf-status">${f.active ? '<span class="status-dot active"></span>' : '<span class="status-dot"></span>'}</span>
        </div>`).join('')}</div>
      </div>`;
    }

    if (filtered.length === 0) {
      html += `<div class="empty-state"><div class="empty-icon">${Icons.Document(40)}</div><h3>No documents</h3><p>Upload or watch a folder to get started.</p></div>`;
    } else {
      if (_groupBy !== 'all') {
        const groupKey = { doc_category: 'doc_category', doc_project: 'doc_project', doc_date: 'doc_date' }[_groupBy] || 'doc_category';
        const groups = {};
        for (const d of filtered) { const k = d[groupKey] || 'Other'; (groups[k] = groups[k] || []).push(d); }
        for (const [gname, gdocs] of Object.entries(groups)) {
          html += `<h4 class="section-head">${BrainApp.escapeHtml(gname)}</h4>`;
          html += _renderDocTable(gdocs);
        }
      } else {
        html += _renderDocTable(filtered);
      }
    }

    el.innerHTML = html;

    el.querySelectorAll('[data-view-doc]').forEach(b => {
      b.addEventListener('click', () => _viewDoc(b.dataset.viewDoc));
    });
    el.querySelectorAll('[data-del-doc]').forEach(b => {
      b.addEventListener('click', () => _deleteDoc(b.dataset.delDoc));
    });
  }

  function _fmtSize(bytes) {
    if (!bytes) return '—';
    if (bytes >= 1048576) return `${(bytes / 1048576).toFixed(1)} MB`;
    if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${bytes} B`;
  }

  function _renderDocTable(docs) {
    return `<table class="records-table">
      <thead><tr><th>Name</th><th>Source</th><th>Status</th><th>Size</th><th>Added</th><th></th></tr></thead>
      <tbody>${docs.map(d => {
        const statusCls = { ready: 'badge-success', building: 'badge-warning', processing: 'badge-warning', failed: 'badge-danger', error: 'badge-danger', deleted: 'badge-muted' }[d.status] || 'badge-muted';
        return `<tr>
          <td class="key-cell">${BrainApp.escapeHtml(d.original_name || '')}</td>
          <td><span class="badge badge-muted">${BrainApp.escapeHtml(d.source_type || '')}</span></td>
          <td><span class="badge ${statusCls}">${BrainApp.escapeHtml(d.status || '')}</span></td>
          <td>${_fmtSize(d.file_size_bytes)}</td>
          <td>${BrainApp.escapeHtml(d.created_at || '')}</td>
          <td class="row-actions">
            <button class="btn btn-sm btn-secondary" data-view-doc="${d.id}">View</button>
            <button class="btn btn-sm btn-danger" data-del-doc="${d.id}">${Icons.Trash(12)}</button>
          </td>
        </tr>`;
      }).join('')}</tbody>
    </table>`;
  }

  async function _viewDoc(id) {
    try {
      const res = await BrainApp.apiFetch(`/documents/${id}`);
      if (!res.ok) { BrainApp.showToast('Failed to load document', 'error'); return; }
      const data = await res.json();
      const doc = data.item || data;
      const overlay = document.createElement('div');
      overlay.className = 'modal-overlay';
      overlay.innerHTML = `<div class="modal modal-lg">
        <div class="modal-header">
          <h3>${BrainApp.escapeHtml(doc.original_name || 'Document')}</h3>
          <button class="btn-close" id="closeDocView">${Icons.Close(16)}</button>
        </div>
        <div class="modal-body"><pre class="doc-content">${BrainApp.escapeHtml(doc.summary || 'No content available.')}</pre></div>
      </div>`;
      document.body.appendChild(overlay);
      document.getElementById('closeDocView').addEventListener('click', () => overlay.remove());
      overlay.addEventListener('click', (e) => { if (e.target === overlay) overlay.remove(); });
    } catch { BrainApp.showToast('Network error', 'error'); }
  }

  function _deleteDoc(id) {
    const doc = _docs.find(d => String(d.id) === String(id));
    const overlay = document.createElement('div');
    overlay.className = 'modal-overlay';
    overlay.innerHTML = `<div class="modal modal-sm">
      <div class="modal-header"><h3>Delete Document</h3></div>
      <p class="modal-desc">Delete "${BrainApp.escapeHtml(doc?.original_name || 'this document')}"? This cannot be undone.</p>
      <div class="modal-actions"><button class="btn btn-secondary" id="keepDoc">Cancel</button><button class="btn btn-danger" id="confirmDelDoc">Delete</button></div>
    </div>`;
    document.body.appendChild(overlay);
    document.getElementById('keepDoc').addEventListener('click', () => overlay.remove());
    overlay.addEventListener('click', (e) => { if (e.target === overlay) overlay.remove(); });
    document.getElementById('confirmDelDoc').addEventListener('click', async () => {
      try {
        const res = await BrainApp.apiFetch(`/documents/${id}`, { method: 'DELETE' });
        if (res.ok) { overlay.remove(); BrainApp.showToast('Document deleted', 'success'); _loaded = false; _load(); }
        else { overlay.remove(); BrainApp.showToast('Delete failed', 'error'); }
      } catch { overlay.remove(); BrainApp.showToast('Network error', 'error'); }
    });
  }

  function _uploadDoc() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.pdf,.docx,.md,.txt,.csv,.zip';
    input.addEventListener('change', async () => {
      const file = input.files[0];
      if (!file) return;
      const formData = new FormData();
      formData.append('file', file);
      try {
        const res = await BrainApp.apiFetch('/documents/upload', { method: 'POST', body: formData }, true);
        if (res.ok) { BrainApp.showToast('Document uploaded', 'success'); _loaded = false; _load(); }
        else { const d = await res.json().catch(() => ({})); BrainApp.showToast(d.error || 'Upload failed', 'error'); }
      } catch { BrainApp.showToast('Network error', 'error'); }
    });
    input.click();
  }


  return { mount, unmount };
})();
