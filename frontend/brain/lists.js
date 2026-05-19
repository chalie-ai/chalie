// Lists panel — list CRUD with inline items.
const PanelLists = (() => {
  let _root = null;
  let _lists = [];
  let _loaded = false;
  let _expanded = {};

  function mount(root) {
    _root = root;
    _render();
  }

  function unmount() { _root = null; }

  function _render() {
    if (!_root) return;
    _root.innerHTML = `<div class="panel-header">
      <h2>Lists</h2>
      <button class="btn btn-primary" id="newListBtn">${Icons.Plus(14)} New List</button>
    </div>
    <div id="listsContent"><div class="loading">Loading…</div></div>`;

    document.getElementById('newListBtn').addEventListener('click', _createList);
    if (!_loaded) _load();
    else _renderLists();
  }

  async function _load() {
    try {
      const res = await BrainApp.apiFetch('/lists');
      if (!res.ok) throw new Error();
      const data = await res.json();
      _lists = data.items || [];
      _loaded = true;
    } catch { BrainApp.showToast('Failed to load lists', 'error'); }
    _renderLists();
  }

  function _renderLists() {
    const el = document.getElementById('listsContent');
    if (!el) return;
    if (_lists.length === 0) {
      el.innerHTML = `<div class="empty-state"><div class="empty-icon">${Icons.List(40)}</div><h3>No lists</h3><p>Create your first list to get started.</p></div>`;
      return;
    }

    el.innerHTML = _lists.map(list => {
      const items = list.items || [];
      const done = list.items ? items.filter(i => i.checked).length : (list.checked_count || 0);
      const total = list.items ? items.length : (list.item_count || 0);
      const pct = total > 0 ? Math.round((done / total) * 100) : 0;
      const isOpen = _expanded[list.id];

      return `<div class="list-card" data-list-id="${list.id}">
        <div class="list-card-header" data-toggle-list="${list.id}">
          <div class="list-card-title">
            <span class="list-chev">${isOpen ? Icons.ChevronDown(14) : Icons.Chevron(14)}</span>
            <span>${BrainApp.escapeHtml(list.name || '')}</span>
            <span class="list-count">${done}/${total}</span>
          </div>
          <div class="list-card-actions">
            <button class="btn btn-sm btn-secondary" data-rename-list="${list.id}">Rename</button>
            <button class="btn btn-sm btn-danger" data-del-list="${list.id}">Delete</button>
          </div>
        </div>
        ${total > 0 ? `<div class="progress-bar"><div class="progress-fill" style="width:${pct}%"></div></div>` : ''}
        ${isOpen ? `<div class="list-items">
          ${items.map(item => `<label class="list-item" data-item-id="${item.id}" data-list-id="${list.id}">
            <input type="checkbox" ${item.checked ? 'checked' : ''}>
            <span class="${item.checked ? 'done' : ''}">${BrainApp.escapeHtml(item.content || '')}</span>
          </label>`).join('')}
          <div class="list-add-row">
            <input type="text" class="list-add-input" placeholder="Add item…" data-add-list="${list.id}" maxlength="300">
          </div>
        </div>` : ''}
      </div>`;
    }).join('');

    el.querySelectorAll('[data-toggle-list]').forEach(h => {
      h.addEventListener('click', (e) => {
        if (e.target.closest('button')) return;
        const id = h.dataset.toggleList;
        _expanded[id] = !_expanded[id];
        if (_expanded[id]) _fetchListDetail(id);
        else _renderLists();
      });
    });

    el.querySelectorAll('.list-item input[type="checkbox"]').forEach(cb => {
      cb.addEventListener('change', () => {
        const label = cb.closest('.list-item');
        _toggleItem(label.dataset.listId, label.dataset.itemId, cb.checked);
      });
    });

    el.querySelectorAll('.list-add-input').forEach(inp => {
      inp.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && inp.value.trim()) {
          _addItem(inp.dataset.addList, inp.value.trim());
          inp.value = '';
        }
      });
    });

    el.querySelectorAll('[data-rename-list]').forEach(b => {
      b.addEventListener('click', (e) => { e.stopPropagation(); _renameList(b.dataset.renameList); });
    });
    el.querySelectorAll('[data-del-list]').forEach(b => {
      b.addEventListener('click', (e) => { e.stopPropagation(); _deleteList(b.dataset.delList); });
    });
  }

  async function _fetchListDetail(listId) {
    try {
      const res = await BrainApp.apiFetch(`/lists/${listId}`);
      if (!res.ok) throw new Error();
      const data = await res.json();
      const detail = data.item || data;
      const idx = _lists.findIndex(l => String(l.id) === String(listId));
      if (idx !== -1) _lists[idx] = { ..._lists[idx], items: detail.items || [] };
    } catch { BrainApp.showToast('Failed to load list items', 'error'); }
    _renderLists();
  }

  async function _toggleItem(listId, itemId, checked) {
    try {
      const list = _lists.find(l => String(l.id) === String(listId));
      if (!list) return;
      const item = (list.items || []).find(i => String(i.id) === String(itemId));
      if (!item) return;
      const endpoint = checked ? 'check' : 'uncheck';
      await BrainApp.apiFetch(`/lists/${listId}/items/${endpoint}`, { method: 'PUT', body: JSON.stringify({ items: [item.content] }) });
      item.checked = checked;
      _renderLists();
    } catch { BrainApp.showToast('Failed to update item', 'error'); }
  }

  async function _addItem(listId, text) {
    try {
      const res = await BrainApp.apiFetch(`/lists/${listId}/items`, { method: 'POST', body: JSON.stringify({ items: [text] }) });
      if (res.ok) await _fetchListDetail(listId);
    } catch { BrainApp.showToast('Failed to add item', 'error'); }
  }

  function _createList() {
    const overlay = document.createElement('div');
    overlay.className = 'modal-overlay';
    overlay.innerHTML = `<div class="modal modal-sm">
      <div class="modal-header"><h3>New List</h3><button class="btn-close" id="closeNewList">${Icons.Close(16)}</button></div>
      <form id="newListForm">
        <div class="form-group"><label for="newListName">List Name</label><input type="text" id="newListName" maxlength="200" placeholder="e.g. Shopping List" required></div>
        <div class="form-actions"><button type="button" class="btn btn-secondary" id="cancelNewList">Cancel</button><button type="submit" class="btn btn-primary">Create</button></div>
      </form>
    </div>`;
    document.body.appendChild(overlay);
    document.getElementById('closeNewList').addEventListener('click', () => overlay.remove());
    document.getElementById('cancelNewList').addEventListener('click', () => overlay.remove());
    overlay.addEventListener('click', (e) => { if (e.target === overlay) overlay.remove(); });
    document.getElementById('newListForm').addEventListener('submit', async (e) => {
      e.preventDefault();
      try {
        const res = await BrainApp.apiFetch('/lists', { method: 'POST', body: JSON.stringify({ name: document.getElementById('newListName').value.trim() }) });
        if (res.ok) { overlay.remove(); BrainApp.showToast('List created', 'success'); _loaded = false; _load(); }
        else BrainApp.showToast('Failed to create list', 'error');
      } catch { BrainApp.showToast('Network error', 'error'); }
    });
  }

  function _renameList(id) {
    const list = _lists.find(l => String(l.id) === String(id));
    const overlay = document.createElement('div');
    overlay.className = 'modal-overlay';
    overlay.innerHTML = `<div class="modal modal-sm">
      <div class="modal-header"><h3>Rename List</h3><button class="btn-close" id="closeRename">${Icons.Close(16)}</button></div>
      <form id="renameForm">
        <div class="form-group"><label for="renameInput">New Name</label><input type="text" id="renameInput" maxlength="200" value="${BrainApp.escapeHtml(list?.name || '')}" required></div>
        <div class="form-actions"><button type="button" class="btn btn-secondary" id="cancelRename">Cancel</button><button type="submit" class="btn btn-primary">Rename</button></div>
      </form>
    </div>`;
    document.body.appendChild(overlay);
    document.getElementById('closeRename').addEventListener('click', () => overlay.remove());
    document.getElementById('cancelRename').addEventListener('click', () => overlay.remove());
    overlay.addEventListener('click', (e) => { if (e.target === overlay) overlay.remove(); });
    document.getElementById('renameForm').addEventListener('submit', async (e) => {
      e.preventDefault();
      try {
        const res = await BrainApp.apiFetch(`/lists/${id}/rename`, { method: 'PUT', body: JSON.stringify({ name: document.getElementById('renameInput').value.trim() }) });
        if (res.ok) { overlay.remove(); BrainApp.showToast('List renamed', 'success'); _loaded = false; _load(); }
        else BrainApp.showToast('Failed to rename', 'error');
      } catch { BrainApp.showToast('Network error', 'error'); }
    });
  }

  async function _deleteList(id) {
    const list = _lists.find(l => String(l.id) === String(id));
    const overlay = document.createElement('div');
    overlay.className = 'modal-overlay';
    overlay.innerHTML = `<div class="modal modal-sm">
      <div class="modal-header"><h3>Delete List</h3></div>
      <p class="modal-desc">Delete "${BrainApp.escapeHtml(list?.name)}"?</p>
      <div class="modal-actions"><button class="btn btn-secondary" id="keepList">Cancel</button><button class="btn btn-danger" id="delList">Delete</button></div>
    </div>`;
    document.body.appendChild(overlay);
    document.getElementById('keepList').addEventListener('click', () => overlay.remove());
    overlay.addEventListener('click', (e) => { if (e.target === overlay) overlay.remove(); });
    document.getElementById('delList').addEventListener('click', async () => {
      try {
        const res = await BrainApp.apiFetch(`/lists/${id}`, { method: 'DELETE' });
        if (res.ok) { overlay.remove(); BrainApp.showToast('List deleted', 'success'); _loaded = false; _load(); }
        else BrainApp.showToast('Delete failed', 'error');
      } catch { BrainApp.showToast('Network error', 'error'); }
    });
  }

  return { mount, unmount };
})();
