// ==========================================
// Configuration
// ==========================================
// If running via HTTP/HTTPS (served by nginx), use relative paths (same host)
// If running as file://, use stored backend host or ask user
let API_BASE = (() => {
    if (window.location.protocol === 'http:' || window.location.protocol === 'https:') {
        // Running via web server — use relative paths
        return '';
    } else {
        // Running as file:// — need full URL
        return localStorage.getItem('chalie_backend_host') || 'http://localhost:8080';
    }
})();

// ==========================================
// State
// ==========================================
let providers = [];
let selectedProviderId = null;  // globally selected provider
let editPlatform = 'ollama';
let editingProviderId = null;
let deletingProviderId = null;
let editModel = '';             // single model for the provider form

// Cognition observability state
let obsData = {};               // cached API responses keyed by subtab name
let obsLoaded = {};             // whether a subtab has been fetched
let activeSubtab = 'memory';    // currently active cognition sub-tab

// ==========================================
// Platform Config
// ==========================================
const PLATFORM_CONFIG = {
    ollama: {
        desc: 'Run locally — no API key needed. Download from <a href="https://ollama.ai" target="_blank">ollama.ai</a>',
        hasHost: true,
        hasApiKey: false,
        modelPlaceholder: 'e.g. gemma4:31b',
        models: [],
    },
    anthropic: {
        desc: 'API key from <a href="https://console.anthropic.com/settings/keys" target="_blank">console.anthropic.com/settings/keys</a>',
        hasHost: false,
        hasApiKey: true,
        modelPlaceholder: 'e.g. claude-sonnet-4-6',
        models: [],
    },
    openai: {
        desc: 'API key from <a href="https://platform.openai.com/api-keys" target="_blank">platform.openai.com/api-keys</a>',
        hasHost: false,
        hasApiKey: true,
        modelPlaceholder: 'e.g. gpt-4o',
        models: ['gpt-4o', 'gpt-4.1', 'o3', 'o4-mini'],
    },
    gemini: {
        desc: 'Free tier available — API key from <a href="https://aistudio.google.com/apikey" target="_blank">aistudio.google.com/apikey</a>',
        hasHost: false,
        hasApiKey: true,
        modelPlaceholder: 'e.g. gemini-2.5-flash',
        models: ['gemini-3.1-pro-preview', 'gemini-3-flash-preview', 'gemini-3.1-flash-lite', 'gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.5-flash-lite'],
    },
    openai_compatible: {
        desc: 'Any OpenAI-compatible API — MiniMax, Groq, DeepSeek, Together, OpenRouter, LM Studio, vLLM. Supply the provider\'s base URL and API key.',
        hasHost: true,
        hasApiKey: true,
        modelPlaceholder: 'e.g. MiniMax-M2',
        models: [],
    },
};

// ==========================================
// API Helpers
// ==========================================
async function apiFetch(path, options = {}, isMultipart = false) {
    // Build full URL: if API_BASE is empty (running via nginx), path is already correct
    // If API_BASE has a value (file:// mode), prepend it
    const url = API_BASE ? `${API_BASE.replace(/\/$/, '')}${path}` : path;
    const headers = {
        ...(isMultipart ? {} : { 'Content-Type': 'application/json' }),
        ...(options.headers || {}),
    };
    const response = await fetch(url, { ...options, headers, credentials: 'same-origin' });
    return response;
}

// ==========================================
// Toast
// ==========================================
function showToast(message, type = 'info', options = {}) {
    const { action, onAction, duration = 3000 } = options;
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;

    if (action && typeof onAction === 'function') {
        const btn = document.createElement('button');
        btn.className = 'toast-action';
        btn.textContent = action;
        btn.addEventListener('click', () => {
            onAction();
            toast.remove();
        });
        toast.appendChild(btn);
    }

    container.appendChild(toast);

    const dismiss = () => {
        toast.style.opacity = '0';
        toast.style.transition = 'opacity 0.3s';
        setTimeout(() => toast.remove(), 300);
    };
    setTimeout(dismiss, duration);
}

// ==========================================
// Init
// ==========================================
async function init() {
    const gate = await globalThis.chalieGateReady;
    if (!gate.stay) return;

    if (gate.providersOnly) applyProvidersOnlyMode();

    try {
        await loadData();
        startSessionHeartbeat();
    } catch (err) {
        console.warn('[brain/app] failed to load initial data:', err);
        showToast('Cannot connect to backend. Is the API running?', 'error');
    }
}

// ==========================================
// Providers-only mode — no providers configured yet, hide everything else.
// ==========================================
function applyProvidersOnlyMode() {
    document.querySelectorAll('#mainTabs .nav-link[data-tab]').forEach(btn => {
        if (btn.dataset.tab !== 'providers') btn.style.display = 'none';
    });
    document.querySelectorAll('.tab-panel').forEach(panel => {
        if (panel.id !== 'tab-providers') panel.style.display = 'none';
    });
    showToast('Configure a provider to start using Chalie.', 'info', { duration: 60 * 60 * 1000 });
}

// ==========================================
// Session heartbeat
// ==========================================
let _sessionHeartbeatInterval = null;
let _sessionHeartbeatFired = false;

async function checkSessionAlive() {
    if (_sessionHeartbeatFired) return;
    try {
        const statusUrl = API_BASE ? `${API_BASE.replace(/\/$/, '')}/auth/status` : '/auth/status';
        const res = await fetch(statusUrl, { credentials: 'same-origin' });
        if (!res.ok) return; // transient — leave the user alone
        const data = await res.json();
        // /auth/status forces has_session=false when vault_state==='locked',
        // so this single check handles both session expiry and vault-locked-
        // after-restart. Only redirect if an account still exists.
        if (data.has_master_account && !data.has_session) {
            _sessionHeartbeatFired = true;
            if (_sessionHeartbeatInterval) {
                clearInterval(_sessionHeartbeatInterval);
                _sessionHeartbeatInterval = null;
            }
            window.location.replace('/login/?next=/brain/');
        }
    } catch (e) { console.warn('[brain/app] session heartbeat check failed:', e); }
}

function startSessionHeartbeat() {
    if (_sessionHeartbeatInterval) return;
    _sessionHeartbeatInterval = setInterval(checkSessionAlive, 5 * 60 * 1000);
    // Re-check whenever the tab becomes visible again so a user returning
    // after a long break gets bounced to login immediately.
    document.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'visible') checkSessionAlive();
    });
}


async function loadData() {
    try {
        const res = await apiFetch('/providers');
        if (res.ok) {
            const data = await res.json();
            providers = data.providers || [];
        } else if (res.status === 401) {
            // Session expired — redirect to login
            window.location.replace('/login/?next=/brain/');
            return;
        } else {
            showToast('Failed to load providers', 'error');
        }
    } catch (err) {
        console.warn('[brain/app] failed to load providers:', err);
        showToast('Failed to connect to backend', 'error');
        return;
    }

    // Fetch selected provider
    try {
        const selRes = await apiFetch('/providers/selected');
        if (selRes.ok) {
            const selData = await selRes.json();
            selectedProviderId = selData.provider ? selData.provider.id : null;
        }
    } catch (e) {
        console.warn('[brain/app] failed to fetch selected provider:', e);
    }

    renderMain();
}

// ==========================================
// Platform Config (Edit Modal Only)
// ==========================================
function selectPlatform(platform, context) {
    const config = PLATFORM_CONFIG[platform];
    editPlatform = platform;

    // Update tab active state
    document.getElementById('editPlatformTabs').querySelectorAll('.platform-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.platform === platform);
    });

    // Update description
    document.getElementById('editPlatformDesc').innerHTML = config.desc;

    // Show/hide host field
    document.getElementById('editHostGroup').style.display = config.hasHost ? '' : 'none';

    // Host field label differs by platform type
    const hostLabel = document.querySelector('#editHostGroup label');
    const hostInput = document.getElementById('editHost');
    if (platform === 'ollama') {
        if (hostLabel) hostLabel.textContent = 'Host';
        hostInput.placeholder = 'http://localhost:11434';
    } else {
        if (hostLabel) hostLabel.textContent = 'Base URL';
        hostInput.placeholder = 'https://api.minimax.io/v1';
    }

    // Show/hide api key field
    document.getElementById('editApiKeyGroup').style.display = config.hasApiKey ? '' : 'none';

    // Update model select
    populateModelSelect();

    // Auto-fetch models on tab switch if the relevant credential is already filled.
    // Skip Gemini (hardcoded list) and OpenAI-compatible (no discovery endpoint).
    // Skip if a fetch is already in-flight to avoid duplicate requests.
    if (!_editModelsFetchInFlight) {
        if (platform === 'ollama') {
            const host = document.getElementById('editHost').value.trim();
            if (host) refreshEditModels();
        } else if (platform === 'openai' || platform === 'anthropic') {
            const key = document.getElementById('editApiKey').value.trim();
            if (key) refreshEditModels();
        }
    }
}

// Set the inline status under the model dropdown.
//   tone ∈ {null, 'loading', 'ok', 'err'}
function setModelStatus(text, tone) {
    const el = document.getElementById('editModelStatus');
    if (!el) return;
    el.textContent = text || '';
    el.className = 'model-status' + (tone ? ' status-' + tone : '');
}

async function fetchProviderModels(platform, credentials) {
    const body = { platform, ...credentials };
    const res = await apiFetch('/providers/list-models', { method: 'POST', body: JSON.stringify(body) });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
        throw new Error(data.error || `Request failed (${res.status})`);
    }
    if (data.error) {
        throw new Error(data.error);
    }
    // Backend returns [{id, display_name}, ...]; collapse to plain ID strings.
    return (data.models || []).map(m => (typeof m === 'string' ? m : (m && m.id) || '')).filter(Boolean);
}



// ==========================================
// Main Render
// ==========================================
function renderMain() {
    document.getElementById('mainContent').style.display = '';
    document.getElementById('mainTabs').style.display = '';
    renderProviders();
}

// ==========================================
// Tab switching
// ==========================================
document.getElementById('mainTabs').addEventListener('click', (e) => {
    const tab = e.target.closest('.nav-link');
    if (!tab) return;
    const tabName = tab.dataset.tab;
    document.querySelectorAll('#mainTabs .nav-link').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    document.getElementById(`tab-${tabName}`).classList.add('active');
    const navEl = document.getElementById('mainNav');
    if (navEl.classList.contains('show')) bootstrap.Collapse.getOrCreateInstance(navEl).hide();

    // Load content when tabs are clicked
    if (tabName === 'cognition') {
        loadCognitionSubtab(activeSubtab);
    } else if (tabName === 'scheduler') {
        loadScheduler();
    } else if (tabName === 'lists') {
        loadLists();
    } else if (tabName === 'documents') {
        loadDocuments();
    } else if (tabName === 'capabilities') {
        loadCapabilities();
    }
});

// ==========================================
// Providers Tab
// ==========================================
function renderProviders() {
    const el = document.getElementById('providersList');
    if (providers.length === 0) {
        el.innerHTML = '<div class="empty-state"><h3>No providers</h3><p>Add your first LLM provider to get started.</p></div>';
        return;
    }
    el.innerHTML = providers.map(p => {
        const modelDisplay = p.model ? escapeHtml(p.model) : 'no model';
        const isSelected = p.id === selectedProviderId;
        const warnIcon = p.decrypt_failed
            ? `<span class="provider-warn" title="Provider is misconfigured. Re-enter credentials">&#9888;</span>`
            : '';
        return `
        <div class="provider-card" data-id="${p.id}">
            <label class="provider-select-radio">
                <input type="radio" name="selected_provider" value="${p.id}" ${isSelected ? 'checked' : ''}>
                <span class="radio-label">${isSelected ? 'Active' : 'Select'}</span>
            </label>
            <div class="provider-info">
                <div class="provider-name">${escapeHtml(p.name)}${warnIcon}</div>
                <div class="provider-meta">
                    <span class="provider-platform-badge badge-${escapeHtml(p.platform)}">${escapeHtml(p.platform)}</span>
                    ${modelDisplay}
                    ${p.host ? ` · ${escapeHtml(p.host)}` : ''}
                </div>
            </div>
            <div class="provider-actions">
                <button class="btn btn-secondary" data-edit-id="${p.id}">Edit</button>
                <button class="btn btn-danger" data-delete-id="${p.id}" data-delete-name="${escapeHtml(p.name)}">Delete</button>
            </div>
        </div>
    `;
    }).join('');

    // Wire delete/edit buttons via data attributes — never interpolate user data into JS strings.
    el.querySelectorAll('[data-edit-id]').forEach(btn => {
        btn.addEventListener('click', () => openEditModal(Number(btn.dataset.editId)));
    });
    el.querySelectorAll('[data-delete-id]').forEach(btn => {
        btn.addEventListener('click', () => confirmDelete(Number(btn.dataset.deleteId), btn.dataset.deleteName));
    });

    // Wire selection radios
    el.querySelectorAll('input[name="selected_provider"]').forEach(radio => {
        radio.addEventListener('change', () => {
            const id = Number(radio.value);
            selectProvider(id);
        });
    });
}

async function selectProvider(id) {
    try {
        const res = await apiFetch('/providers/selected', {
            method: 'PUT',
            body: JSON.stringify({ provider_id: id }),
        });
        if (res.ok) {
            selectedProviderId = id;
            renderProviders();
            showToast('Provider selected', 'success');
        } else {
            const err = await res.json();
            showToast(err.error || 'Failed to select provider', 'error');
            renderProviders(); // revert UI
        }
    } catch (e) {
        console.warn('[brain/app] failed to select provider:', e);
        showToast('Failed to select provider', 'error');
        renderProviders(); // revert UI
    }
}

document.getElementById('addProviderBtn').addEventListener('click', () => {
    openEditModal(null);
});

function openEditModal(id) {
    editingProviderId = id;
    const modal = document.getElementById('providerModal');
    document.getElementById('providerModalTitle').textContent = id ? 'Edit Provider' : 'Add Provider';
    document.getElementById('editProviderId').value = id || '';

    // Reset form
    document.getElementById('providerForm').reset();
    editModel = '';

    if (id) {
        const p = providers.find(x => x.id === id);
        if (p) {
            editPlatform = p.platform;
            document.getElementById('editName').value = p.name;
            editModel = p.model || '';
            if (p.host) document.getElementById('editHost').value = p.host;
        }
    } else {
        editPlatform = 'ollama';
    }

    populateModelSelect();

    // Clear any previous test result
    const testResult = document.getElementById('testResult');
    testResult.className = 'test-result hidden';
    testResult.innerHTML = '';

    selectPlatform(editPlatform, 'edit');
    modal.classList.remove('hidden');

    // Auto-load model list when editing an existing provider so the dropdown
    // isn't empty on open. For new providers credential fields are blank so
    // the tab-switch fetch inside selectPlatform won't fire (by design).
    if (id) {
        refreshEditModels();
    }
}

function populateModelSelect() {
    const select = document.getElementById('editModel');
    if (!select) return;
    select.innerHTML = '<option value="">Select model...</option>';

    const config = PLATFORM_CONFIG[editPlatform];
    if (config.models && config.models.length > 0) {
        config.models.forEach(m => {
            const opt = document.createElement('option');
            opt.value = m;
            opt.textContent = m;
            if (m === editModel) opt.selected = true;
            select.appendChild(opt);
        });
    }
    // If no curated list and we have a value, add it as an option
    if (editModel && !Array.from(select.options).some(o => o.value === editModel)) {
        const opt = document.createElement('option');
        opt.value = editModel;
        opt.textContent = editModel;
        opt.selected = true;
        select.appendChild(opt);
    }
}

document.getElementById('closeProviderModal').addEventListener('click', () => {
    document.getElementById('providerModal').classList.add('hidden');
});

document.getElementById('cancelProviderBtn').addEventListener('click', () => {
    document.getElementById('providerModal').classList.add('hidden');
});

// Platform tabs in edit modal
document.getElementById('editPlatformTabs').addEventListener('click', (e) => {
    const tab = e.target.closest('.platform-tab');
    if (tab) {
        selectPlatform(tab.dataset.platform, 'edit');
        const testResult = document.getElementById('testResult');
        testResult.className = 'test-result hidden';
        testResult.innerHTML = '';
    }
});

// Live model fetch — single guarded code path for all platforms.
// Guards against duplicate in-flight requests (tab-switch + keyup can both fire).
let _editModelsFetchInFlight = false;
async function refreshEditModels() {
    if (_editModelsFetchInFlight) return;

    const platform = editPlatform;

    let credentials;
    if (platform === 'ollama') {
        const host = document.getElementById('editHost').value.trim();
        credentials = { host: host || 'http://localhost:11434' };
    } else if (platform === 'openai' || platform === 'anthropic') {
        const key = document.getElementById('editApiKey').value.trim();
        if (!key) return;
        credentials = { api_key: key };
    } else {
        return;
    }

    const select = document.getElementById('editModel');
    _editModelsFetchInFlight = true;
    if (select) select.disabled = true;
    setModelStatus('Loading models…', 'loading');

    try {
        const names = await fetchProviderModels(platform, credentials);
        populateModelSelectFromFetch(names);
        if (names.length === 0) {
            setModelStatus(
                platform === 'ollama'
                    ? 'No models installed on this Ollama host'
                    : 'No models returned',
                'err',
            );
        } else {
            setModelStatus(
                `${names.length} model${names.length === 1 ? '' : 's'} available`,
                'ok',
            );
        }
    } catch (e) {
        setModelStatus(e.message || 'Could not fetch models', 'err');
    } finally {
        if (select) select.disabled = false;
        _editModelsFetchInFlight = false;
    }
}

// Populate the dropdown from a freshly-fetched list. Preserves the user's
// current selection if the model still exists in the new list; otherwise
// appends it as a trailing option so the existing provider config isn't lost.
function populateModelSelectFromFetch(names) {
    const select = document.getElementById('editModel');
    if (!select) return;
    const prev = editModel || select.value;
    select.innerHTML = '<option value="">Select model...</option>';
    names.forEach(n => {
        const opt = document.createElement('option');
        opt.value = n;
        opt.textContent = n;
        if (n === prev) opt.selected = true;
        select.appendChild(opt);
    });
    if (prev && !names.includes(prev)) {
        const opt = document.createElement('option');
        opt.value = prev;
        opt.textContent = prev;
        opt.selected = true;
        select.appendChild(opt);
    }
}

// Debounced keyup on host (ollama) and api_key (openai/anthropic). 600 ms is
// comfortably above typical inter-keystroke latency so a long paste collapses
// to a single fetch while still feeling live.
const _MODEL_FETCH_DEBOUNCE_MS = 600;
let _modelFetchTimer = null;
function debouncedRefreshModels() {
    clearTimeout(_modelFetchTimer);
    _modelFetchTimer = setTimeout(refreshEditModels, _MODEL_FETCH_DEBOUNCE_MS);
}
document.getElementById('editHost').addEventListener('keyup', () => {
    if (editPlatform === 'ollama') debouncedRefreshModels();
});
document.getElementById('editApiKey').addEventListener('keyup', () => {
    if (editPlatform === 'openai' || editPlatform === 'anthropic') debouncedRefreshModels();
});

// Full provider test (all platforms)
document.getElementById('testProviderBtn').addEventListener('click', async () => {
    const btn = document.getElementById('testProviderBtn');
    const resultEl = document.getElementById('testResult');

    btn.disabled = true;
    btn.textContent = 'Testing…';
    resultEl.className = 'test-result';
    resultEl.innerHTML = '<span style="color:var(--text-muted)">Testing connection…</span>';

    const id = editingProviderId;
    const platform = editPlatform;
    const config = PLATFORM_CONFIG[platform];

    const testModel = document.getElementById('editModel').value.trim();
    const body = {
        platform,
        model: testModel,
    };

    if (id) body.provider_id = id;
    if (config.hasHost) {
        const hostVal = document.getElementById('editHost').value.trim();
        body.host = hostVal || (platform === 'ollama' ? 'http://localhost:11434' : '');
    }
    if (config.hasApiKey) {
        const key = document.getElementById('editApiKey').value.trim();
        if (key) body.api_key = key;
    }

    try {
        const res = await apiFetch('/providers/test', { method: 'POST', body: JSON.stringify(body) });
        const data = await res.json();

        if (data.success) {
            const latency = data.latency_ms ? ` · ${data.latency_ms}ms` : '';
            resultEl.className = 'test-result test-success';
            resultEl.innerHTML = `✓ ${escapeHtml(data.message || 'Connected')}${latency}`;
        } else {
            resultEl.className = 'test-result test-error';
            let html = `✗ ${escapeHtml(data.error || 'Connection failed')}`;
            if (data.hint) {
                html += `<div class="test-hint">${escapeHtml(data.hint)}</div>`;
            }
            resultEl.innerHTML = html;
        }
    } catch (e) {
        console.warn('[brain/app] provider connection test failed:', e);
        resultEl.className = 'test-result test-error';
        resultEl.innerHTML = '✗ Could not reach the backend';
    } finally {
        btn.disabled = false;
        btn.textContent = 'Test Connection';
    }
});

document.getElementById('providerForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const id = editingProviderId;
    const platform = editPlatform;
    const config = PLATFORM_CONFIG[platform];

    const modelVal = document.getElementById('editModel').value.trim();
    if (!modelVal) {
        showToast('Select a model', 'error');
        return;
    }
    editModel = modelVal;

    const body = {
        name: document.getElementById('editName').value.trim(),
        platform: platform,
        model: editModel,
    };

    if (config.hasHost) {
        const hostVal = document.getElementById('editHost').value.trim();
        body.host = hostVal || (platform === 'ollama' ? 'http://localhost:11434' : '');
    }
    if (config.hasApiKey) {
        const key = document.getElementById('editApiKey').value.trim();
        if (key) body.api_key = key;
    }

    let res;
    if (id) {
        res = await apiFetch(`/providers/${id}`, { method: 'PUT', body: JSON.stringify(body) });
    } else {
        res = await apiFetch('/providers', { method: 'POST', body: JSON.stringify(body) });
    }

    if (res.ok) {
        const wasFirstAdd = !id && providers.length === 0;
        document.getElementById('providerModal').classList.add('hidden');
        document.getElementById('providerForm').reset();
        editModel = '';
        showToast(id ? 'Provider updated' : 'Provider added', 'success');

        // First-ever provider unlocks the rest of the Brain UI (auth-gate
        // sets providersOnly=!has_providers at page load). A full reload is
        // the only way to re-run the gate and reveal the hidden tabs.
        if (wasFirstAdd) {
            window.location.reload();
            return;
        }

        // Re-fetch from the server so the list reflects canonical state
        // (handles api_key masking, server-side defaults, etc.) and re-render.
        await loadData();
    } else {
        const err = await res.json();
        showToast(err.error || 'Failed to save provider', 'error');
    }
});

// ==========================================
// Delete
// ==========================================
function confirmDelete(id, name) {
    deletingProviderId = id;
    document.getElementById('deleteModalDesc').textContent = `Are you sure you want to delete "${name}"?`;
    document.getElementById('deleteModal').classList.remove('hidden');
}

document.getElementById('cancelDeleteBtn').addEventListener('click', () => {
    document.getElementById('deleteModal').classList.add('hidden');
    deletingProviderId = null;
});

document.getElementById('confirmDeleteBtn').addEventListener('click', async () => {
    if (!deletingProviderId) return;

    if (providers.length <= 1) {
        showToast('At least one provider must remain', 'error');
        document.getElementById('deleteModal').classList.add('hidden');
        return;
    }

    const res = await apiFetch(`/providers/${deletingProviderId}`, { method: 'DELETE' });
    if (res.ok) {
        providers = providers.filter(p => p.id !== deletingProviderId);
        document.getElementById('deleteModal').classList.add('hidden');
        deletingProviderId = null;
        renderProviders();
        showToast('Provider deleted', 'success');
    } else {
        const err = await res.json();
        showToast(err.error || 'Cannot delete provider', 'error');
        document.getElementById('deleteModal').classList.add('hidden');
    }
});

// ==========================================
// Scheduler Tab
// ==========================================
let scheduleItems = [];
let scheduleFilter = 'all';
let scheduleOffset = 0;
let scheduleTotal = 0;
let editingScheduleId = null;
let cancellingScheduleId = null;
const SCHEDULE_LIMIT = 50;

async function loadScheduler(append = false) {
    if (!append) {
        scheduleOffset = 0;
        scheduleItems = [];
        document.getElementById('schedulerList').innerHTML = '<div class="loading">Loading schedule…</div>';
    }

    try {
        const params = new URLSearchParams({
            status: scheduleFilter,
            limit: SCHEDULE_LIMIT,
            offset: scheduleOffset,
        });
        const res = await apiFetch(`/scheduler?${params}`);
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            showToast(err.error || 'Failed to load schedule', 'error');
            return;
        }
        const data = await res.json();
        scheduleItems = append ? [...scheduleItems, ...data.items] : data.items;
        scheduleTotal = data.total;
        scheduleOffset = scheduleItems.length;
        renderScheduler();
    } catch (e) {
        document.getElementById('schedulerList').innerHTML =
            `<div class="empty-state"><h3>Error loading schedule</h3><p>${escapeHtml(e.message)}</p></div>`;
    }
}

function renderScheduler() {
    const list = document.getElementById('schedulerList');
    const footer = document.getElementById('schedulerFooter');
    const loadMoreBtn = document.getElementById('scheduleLoadMoreBtn');
    const clearBtn = document.getElementById('clearHistoryBtn');

    if (scheduleItems.length === 0) {
        list.innerHTML = `
            <div class="empty-state">
                <h3>Nothing scheduled</h3>
                <p>Create a scheduled reminder or task and Chalie will act on it automatically.</p>
            </div>`;
        footer.classList.add('hidden');
        return;
    }

    if (scheduleFilter === 'all') {
        list.innerHTML = renderAccordionList(scheduleItems);
        list.querySelectorAll('.schedule-accordion-row').forEach(row => {
            row.querySelector('.schedule-accordion-header').addEventListener('click', function (e) {
                if (e.target.closest('button')) return;
                toggleAccordionRow(row);
            });
        });
    } else {
        list.innerHTML = scheduleItems.map(renderScheduleCard).join('');
    }

    // Show/hide footer controls
    footer.classList.remove('hidden');
    loadMoreBtn.classList.toggle('hidden', scheduleItems.length >= scheduleTotal);
    const hasHistory = scheduleItems.some(i => i.status !== 'pending');
    const showClear = (scheduleFilter === 'all' || scheduleFilter !== 'pending') && hasHistory;
    clearBtn.classList.toggle('hidden', !showClear);
}

function renderScheduleCard(item) {
    const msg = item.message || '';
    const truncated = msg.length > 120 ? msg.slice(0, 120) + '…' : msg;
    const due = item.due_at ? new Date(item.due_at).toLocaleString() : '—';
    const lastFired = item.last_fired_at ? new Date(item.last_fired_at).toLocaleString() : null;
    const isPending = item.status === 'pending';

    const statusClass = {
        pending: '--pending',
        fired: '--fired',
        failed: '--failed',
        cancelled: '--cancelled',
    }[item.status] || '';

    const typeBadge = `<span class="schedule-badge --type-${escapeHtml(item.item_type)}">${escapeHtml(item.item_type)}</span>`;
    const recurrBadge = item.recurrence
        ? `<span class="schedule-badge --recurrence">${escapeHtml(formatRecurrence(item.recurrence))}</span>`
        : '';
    const firedInfo = lastFired
        ? `<span class="schedule-card__last-fired">Last fired: ${lastFired}</span>`
        : '';

    const actions = isPending ? `
        <button class="tool-card__btn" data-edit-schedule="${escapeHtml(item.id)}">Edit</button>
        <button class="tool-card__btn --danger" data-cancel-schedule="${escapeHtml(item.id)}">Cancel</button>
    ` : '';

    return `
        <div class="schedule-card">
            <div class="schedule-card__body">
                <div class="schedule-card__message">${escapeHtml(truncated)}</div>
                <div class="schedule-card__meta">
                    <span class="schedule-card__due">${due}</span>
                    ${typeBadge}
                    ${recurrBadge}
                    ${firedInfo}
                </div>
            </div>
            <div class="schedule-card__right">
                <span class="schedule-card__status ${statusClass}">${escapeHtml(item.status)}</span>
                <div class="schedule-card__actions">${actions}</div>
            </div>
        </div>
    `;
}

function toLocalDatetimeString(date) {
    // Formats a Date as "YYYY-MM-DDTHH:MM" in local timezone for datetime-local input
    const pad = n => String(n).padStart(2, '0');
    return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

function updateScheduleFormHints() {
    const recurrence = document.getElementById('scheduleRecurrence').value;
    const dueAtVal = document.getElementById('scheduleDueAt').value;
    const label = document.getElementById('scheduleDueAtLabel');
    const hint = document.getElementById('recurrenceHint');

    // Recurrence-aware label
    label.textContent = recurrence
        ? 'First Occurrence & Recurring Time'
        : 'Due Date & Time';

    // Dynamic pattern hint
    if (!recurrence || !dueAtVal) {
        hint.style.display = 'none';
        hint.textContent = '';
        return;
    }

    const date = new Date(dueAtVal);
    if (Number.isNaN(date.getTime())) {
        hint.style.display = 'none';
        hint.textContent = '';
        return;
    }

    const DAYS = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
    const dayName = DAYS[date.getDay()];
    const timeStr = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const dom = date.getDate();
    let domSuffix;
    if (dom === 1) domSuffix = 'st';
    else if (dom === 2) domSuffix = 'nd';
    else if (dom === 3) domSuffix = 'rd';
    else domSuffix = 'th';

    let pattern = '';
    if (recurrence === 'interval') {
        const mins = Number.parseInt(document.getElementById('scheduleIntervalMinutes').value, 10);
        if (mins >= 1 && mins <= 1440) {
            const h = Math.floor(mins / 60), m = mins % 60;
            const parts = [];
            if (h > 0) parts.push(`${h}h`);
            if (m > 0) parts.push(`${m}m`);
            pattern = `Fires every ${parts.join(' ')} — starting ${timeStr}`;
        }
    } else {
        const patterns = {
            hourly:   'Fires every hour (use active window below to restrict hours)',
            daily:    `Fires every day at ${timeStr}`,
            weekdays: `Fires every weekday (Mon–Fri) at ${timeStr}`,
            weekly:   `Fires every ${dayName} at ${timeStr}`,
            monthly:  `Fires on the ${dom}${domSuffix} of every month at ${timeStr}`,
        };
        pattern = patterns[recurrence] || '';
    }

    if (pattern) {
        hint.textContent = pattern;
        hint.style.display = '';
    } else {
        hint.style.display = 'none';
        hint.textContent = '';
    }
}

function openCreateSchedule() {
    editingScheduleId = null;
    document.getElementById('scheduleModalTitle').textContent = 'New Schedule';
    document.getElementById('scheduleForm').reset();

    // Default due_at = +1 hour from now
    const defaultDue = new Date(Date.now() + 60 * 60 * 1000);
    document.getElementById('scheduleDueAt').value = toLocalDatetimeString(defaultDue);
    document.getElementById('windowGroup').style.display = 'none';
    document.getElementById('intervalGroup').style.display = 'none';
    updateScheduleFormHints();

    document.getElementById('scheduleModal').classList.remove('hidden');
    document.getElementById('scheduleMessage').focus();
}

function openEditSchedule(id) {
    const item = scheduleItems.find(i => i.id === id);
    if (!item) return;

    editingScheduleId = id;
    document.getElementById('scheduleModalTitle').textContent = 'Edit Schedule';
    document.getElementById('scheduleMessage').value = item.message || '';
    document.getElementById('scheduleDueAt').value = item.due_at
        ? toLocalDatetimeString(new Date(item.due_at))
        : '';
    document.getElementById('scheduleType').value = item.item_type || 'notification';

    // Decode interval:N recurrence
    const rawRec = item.recurrence || '';
    let displayRec = rawRec;
    if (rawRec.startsWith('interval:')) {
        displayRec = 'interval';
        document.getElementById('scheduleIntervalMinutes').value = rawRec.split(':')[1] || '30';
    }
    document.getElementById('scheduleRecurrence').value = displayRec;
    document.getElementById('scheduleWindowStart').value = item.window_start || '';
    document.getElementById('scheduleWindowEnd').value = item.window_end || '';
    document.getElementById('windowGroup').style.display = displayRec === 'hourly' ? '' : 'none';
    document.getElementById('intervalGroup').style.display = displayRec === 'interval' ? '' : 'none';
    updateScheduleFormHints();

    document.getElementById('scheduleModal').classList.remove('hidden');
    document.getElementById('scheduleMessage').focus();
}

document.getElementById('scheduleForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const localValue = document.getElementById('scheduleDueAt').value;
    if (!localValue) {
        showToast('Due date is required', 'error');
        return;
    }

    // Encode interval:N
    let recurrenceValue = document.getElementById('scheduleRecurrence').value || null;
    if (recurrenceValue === 'interval') {
        const mins = Number.parseInt(document.getElementById('scheduleIntervalMinutes').value, 10);
        if (!mins || mins < 1 || mins > 1440) {
            showToast('Interval must be between 1 and 1440 minutes', 'error');
            return;
        }
        recurrenceValue = `interval:${mins}`;
    }

    const body = {
        message: document.getElementById('scheduleMessage').value.trim(),
        due_at: new Date(localValue).toISOString(),
        item_type: document.getElementById('scheduleType').value,
        recurrence: recurrenceValue,
        window_start: document.getElementById('scheduleWindowStart').value || null,
        window_end: document.getElementById('scheduleWindowEnd').value || null,
    };

    const btn = e.target.querySelector('button[type="submit"]');
    btn.disabled = true;
    btn.textContent = 'Saving…';

    try {
        let res;
        if (editingScheduleId) {
            res = await apiFetch(`/scheduler/${editingScheduleId}`, { method: 'PUT', body: JSON.stringify(body) });
        } else {
            res = await apiFetch('/scheduler', { method: 'POST', body: JSON.stringify(body) });
        }

        const data = await res.json();
        if (res.ok) {
            document.getElementById('scheduleModal').classList.add('hidden');
            showToast(editingScheduleId ? 'Schedule updated' : 'Schedule created', 'success');
            await loadScheduler();
        } else {
            showToast(data.error || 'Failed to save schedule', 'error');
        }
    } catch (err) {
        console.warn('[brain/app] failed to save schedule:', err);
        showToast('Network error', 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = 'Save';
    }
});

function confirmCancelSchedule(id) {
    cancellingScheduleId = id;
    const item = scheduleItems.find(i => i.id === id);
    const msg = item ? item.message : '';
    const truncated = msg.length > 80 ? msg.slice(0, 80) + '…' : msg;
    document.getElementById('cancelScheduleDesc').textContent =
        `Cancel: "${truncated}"?`;
    document.getElementById('cancelScheduleModal').classList.remove('hidden');
}

async function executeCancelSchedule() {
    if (!cancellingScheduleId) return;
    const id = cancellingScheduleId;
    document.getElementById('cancelScheduleModal').classList.add('hidden');
    cancellingScheduleId = null;

    try {
        const res = await apiFetch(`/scheduler/${id}`, { method: 'DELETE' });
        if (res.ok) {
            showToast('Schedule cancelled', 'success');
            await loadScheduler();
        } else {
            const err = await res.json().catch(() => ({}));
            showToast(err.error || 'Failed to cancel', 'error');
        }
    } catch (e) {
        console.warn('[brain/app] failed to cancel schedule:', e);
        showToast('Network error', 'error');
    }
}

async function clearHistory() {
    try {
        const res = await apiFetch('/scheduler/history', { method: 'DELETE' });
        const data = await res.json();
        if (res.ok) {
            showToast(`Removed ${data.deleted} item(s)`, 'success');
            await loadScheduler();
        } else {
            showToast(data.error || 'Failed to clear history', 'error');
        }
    } catch (e) {
        console.warn('[brain/app] failed to clear history:', e);
        showToast('Network error', 'error');
    }
}

function formatRecurrence(recurrence) {
    if (!recurrence) return '';
    if (recurrence.startsWith('interval:')) {
        const mins = Number.parseInt(recurrence.split(':')[1], 10);
        if (Number.isNaN(mins)) return recurrence;
        const h = Math.floor(mins / 60), m = mins % 60;
        if (h === 0) return `Every ${m}m`;
        if (m === 0) return `Every ${h}h`;
        return `Every ${h}h ${m}m`;
    }
    return recurrence;
}

function renderAccordionList(items) {
    if (!items.length) return '';
    // Group by group_id; preserve first-seen order
    const groups = new Map();
    for (const item of items) {
        const gid = item.group_id || item.id;
        if (!groups.has(gid)) groups.set(gid, []);
        groups.get(gid).push(item);
    }
    const parts = [];
    for (const [gid, groupItems] of groups) {
        // Representative: pending first, then most recent by due_at
        const rep = groupItems.find(i => i.status === 'pending')
            || [...groupItems].sort((a, b) => new Date(b.due_at) - new Date(a.due_at))[0];
        // Only accordion for recurring items; flat card for one-time
        if (rep.recurrence) {
            parts.push(renderAccordionRow(rep, gid));
        } else {
            parts.push(renderScheduleCard(rep));
        }
    }
    return parts.join('');
}

function renderAccordionRow(item, groupId) {
    const msg = item.message || '';
    const truncated = msg.length > 120 ? msg.slice(0, 120) + '…' : msg;
    const due = item.due_at ? new Date(item.due_at).toLocaleString() : '—';
    const isPending = item.status === 'pending';
    const statusClass = { pending: '--pending', fired: '--fired', failed: '--failed', cancelled: '--cancelled' }[item.status] || '';
    const typeBadge = `<span class="schedule-badge --type-${escapeHtml(item.item_type)}">${escapeHtml(item.item_type)}</span>`;
    const recurrBadge = `<span class="schedule-badge --recurrence">${escapeHtml(formatRecurrence(item.recurrence))}</span>`;
    const actions = isPending ? `
        <button class="tool-card__btn" data-edit-schedule="${escapeHtml(item.id)}">Edit</button>
        <button class="tool-card__btn --danger" data-cancel-schedule="${escapeHtml(item.id)}">Cancel</button>
    ` : '';
    return `
        <div class="schedule-accordion-row" data-group-id="${escapeHtml(groupId)}" data-loaded="false">
            <div class="schedule-accordion-header">
                <div class="schedule-card__body">
                    <div class="schedule-card__message">${escapeHtml(truncated)}</div>
                    <div class="schedule-card__meta">
                        <span class="schedule-card__due">${isPending ? 'Next:' : 'Last:'} ${due}</span>
                        ${typeBadge}${recurrBadge}
                    </div>
                </div>
                <div class="schedule-card__right">
                    <span class="schedule-card__status ${statusClass}">${escapeHtml(item.status)}</span>
                    <div class="schedule-card__actions">${actions}</div>
                    <span class="accordion-chevron">▸</span>
                </div>
            </div>
            <div class="schedule-accordion-body" style="display:none">
                <p class="form-hint fires-loading">Loading history…</p>
            </div>
        </div>
    `;
}

function toggleAccordionRow(row) {
    const body = row.querySelector('.schedule-accordion-body');
    const chevron = row.querySelector('.accordion-chevron');
    const isOpen = body.style.display !== 'none';
    if (isOpen) {
        body.style.display = 'none';
        chevron.textContent = '▸';
    } else {
        body.style.display = '';
        chevron.textContent = '▾';
        if (row.dataset.loaded === 'false') {
            row.dataset.loaded = 'true';
            loadGroupFires(row.dataset.groupId, body);
        }
    }
}

async function loadGroupFires(groupId, container) {
    try {
        const res = await apiFetch(`/scheduler/group/${encodeURIComponent(groupId)}`);
        if (!res.ok) {
            container.innerHTML = '<p class="form-hint">Could not load history.</p>';
            return;
        }
        const data = await res.json();
        const items = data.items || [];
        if (items.length === 0) {
            container.innerHTML = '<p class="form-hint">No fire history yet.</p>';
            return;
        }
        container.innerHTML = `<div class="fire-history-list">${
            items.map(item => {
                const d = item.due_at ? new Date(item.due_at).toLocaleString() : '—';
                const sc = { pending: '--pending', fired: '--fired', failed: '--failed', cancelled: '--cancelled' }[item.status] || '';
                return `<div class="fire-history-item">
                    <span class="fire-history-item__status ${sc}">${escapeHtml(item.status)}</span>
                    <span class="fire-history-item__date">${d}</span>
                </div>`;
            }).join('')
        }</div>`;
    } catch (e) {
        console.warn('[brain/app] failed to load schedule history:', e);
        container.innerHTML = '<p class="form-hint">Error loading history.</p>';
    }
}

// Scheduler event listeners
document.getElementById('newScheduleBtn').addEventListener('click', openCreateSchedule);

document.getElementById('closeScheduleModal').addEventListener('click', () => {
    document.getElementById('scheduleModal').classList.add('hidden');
});
document.getElementById('cancelScheduleFormBtn').addEventListener('click', () => {
    document.getElementById('scheduleModal').classList.add('hidden');
});
document.getElementById('keepScheduleBtn').addEventListener('click', () => {
    document.getElementById('cancelScheduleModal').classList.add('hidden');
    cancellingScheduleId = null;
});
document.getElementById('confirmCancelScheduleBtn').addEventListener('click', executeCancelSchedule);

document.getElementById('schedulerList').addEventListener('click', (e) => {
    const editBtn = e.target.closest('[data-edit-schedule]');
    if (editBtn) { openEditSchedule(editBtn.dataset.editSchedule); return; }
    const cancelBtn = e.target.closest('[data-cancel-schedule]');
    if (cancelBtn) { confirmCancelSchedule(cancelBtn.dataset.cancelSchedule); }
});
document.getElementById('scheduleLoadMoreBtn').addEventListener('click', () => loadScheduler(true));
document.getElementById('clearHistoryBtn').addEventListener('click', clearHistory);

document.getElementById('scheduleRecurrence').addEventListener('change', (e) => {
    const val = e.target.value;
    document.getElementById('windowGroup').style.display = val === 'hourly' ? '' : 'none';
    document.getElementById('intervalGroup').style.display = val === 'interval' ? '' : 'none';
    updateScheduleFormHints();
});

document.getElementById('scheduleIntervalMinutes').addEventListener('input', updateScheduleFormHints);

document.getElementById('scheduleDueAt').addEventListener('change', updateScheduleFormHints);

document.querySelector('.scheduler-filters').addEventListener('click', (e) => {
    const btn = e.target.closest('.filter-tab');
    if (!btn) return;
    document.querySelectorAll('.filter-tab').forEach(t => t.classList.remove('active'));
    btn.classList.add('active');
    scheduleFilter = btn.dataset.filter;
    loadScheduler();
});

// ==========================================
// Lists Tab
// ==========================================
let userLists = [];
let expandedListId = null;
let expandedListData = null;
let renamingListId = null;
let deletingListId = null;

async function loadLists() {
    document.getElementById('listsContainer').innerHTML = '<div class="loading">Loading lists…</div>';
    try {
        const res = await apiFetch('/lists');
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            document.getElementById('listsContainer').innerHTML =
                `<div class="empty-state"><h3>Error loading lists</h3><p>${escapeHtml(err.error || 'Unknown error')}</p></div>`;
            return;
        }
        const data = await res.json();
        userLists = data.items || [];
        renderLists();
    } catch (e) {
        document.getElementById('listsContainer').innerHTML =
            `<div class="empty-state"><h3>Error loading lists</h3><p>${escapeHtml(e.message)}</p></div>`;
    }
}

function renderLists() {
    const container = document.getElementById('listsContainer');
    if (!container) return;
    if (userLists.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <h3>No lists yet</h3>
                <p>Create a list to get started, or add items by chatting with Chalie.</p>
            </div>`;
        return;
    }
    container.innerHTML = userLists.map(lst => renderListCard(lst)).join('');
}

function renderListCard(lst) {
    const isExpanded = expandedListId === lst.id;
    const total = lst.item_count || 0;
    const checked = lst.checked_count || 0;
    const pct = total > 0 ? Math.round((checked / total) * 100) : 0;

    const progressHtml = total > 0 ? `
        <div class="list-card__progress-bar">
            <div class="list-card__progress-fill" style="width: ${pct}%"></div>
        </div>` : '';

    let expandedHtml = '';
    if (isExpanded) {
        if (expandedListData === null) {
            expandedHtml = `<div class="list-card__items"><div class="loading" style="padding: 12px 0;">Loading items…</div></div>`;
        } else {
            const items = [...(expandedListData.items || [])].sort((a, b) => {
                if (a.checked === b.checked) return 0;
                return a.checked ? 1 : -1;
            });

            let itemsHtml = '';
            if (items.length === 0) {
                itemsHtml = '<p class="list-card__empty">No items yet. Add something below.</p>';
            } else {
                itemsHtml = items.map(item => `
                    <div class="list-item ${item.checked ? 'list-item--checked' : ''}"
                         data-list-id="${escapeHtml(lst.id)}"
                         data-content="${escapeHtml(item.content)}">
                        <label class="list-item__checkbox">
                            <input type="checkbox" ${item.checked ? 'checked' : ''}>
                            <span class="list-item__check-mark"></span>
                        </label>
                        <span class="list-item__content">${escapeHtml(item.content)}</span>
                        <button class="list-item__remove"
                                data-list-id="${escapeHtml(lst.id)}"
                                data-content="${escapeHtml(item.content)}"
                                title="Remove">✕</button>
                    </div>
                `).join('');
            }

            const addHtml = `
                <div class="list-card__add-item">
                    <input type="text"
                           class="list-card__add-input"
                           data-list-id="${escapeHtml(lst.id)}"
                           id="addItemInput-${escapeHtml(lst.id)}"
                           placeholder="Add item…"
                           maxlength="500">
                    <button class="btn btn-secondary list-card__add-btn"
                            data-list-id="${escapeHtml(lst.id)}">Add</button>
                </div>`;

            expandedHtml = `<div class="list-card__items">${itemsHtml}${addHtml}</div>`;
        }
    }

    return `
        <div class="list-card ${isExpanded ? 'list-card--expanded' : ''}" data-list-id="${escapeHtml(lst.id)}">
            <div class="list-card__header" data-list-id="${escapeHtml(lst.id)}">
                <div class="list-card__title-row">
                    <span class="list-card__name">${escapeHtml(lst.name)}</span>
                    <span class="list-card__count">${total} item${total !== 1 ? 's' : ''}${checked > 0 ? ` · ${checked} done` : ''}</span>
                </div>
                <div class="list-card__header-actions">
                    <button class="tool-card__btn"
                            data-action="rename"
                            data-list-id="${escapeHtml(lst.id)}"
                            data-list-name="${escapeHtml(lst.name)}">Rename</button>
                    <button class="tool-card__btn --danger"
                            data-action="delete"
                            data-list-id="${escapeHtml(lst.id)}"
                            data-list-name="${escapeHtml(lst.name)}">Delete</button>
                </div>
            </div>
            ${progressHtml}
            ${expandedHtml}
        </div>
    `;
}

async function toggleListExpand(id) {
    if (expandedListId === id) {
        expandedListId = null;
        expandedListData = null;
        renderLists();
        return;
    }

    expandedListId = id;
    expandedListData = null;
    renderLists();

    try {
        const res = await apiFetch(`/lists/${id}`);
        if (res.ok) {
            const data = await res.json();
            expandedListData = data.item;
        } else {
            showToast('Failed to load list', 'error');
            expandedListId = null;
        }
    } catch (e) {
        console.warn('[brain/app] failed to load list:', e);
        showToast('Network error', 'error');
        expandedListId = null;
    }
    renderLists();
}

async function refreshExpandedList(id) {
    try {
        const [summaryRes, detailRes] = await Promise.all([
            apiFetch('/lists'),
            apiFetch(`/lists/${id}`),
        ]);

        if (summaryRes.ok) {
            const data = await summaryRes.json();
            userLists = data.items || [];
        }
        if (detailRes.ok) {
            const data = await detailRes.json();
            expandedListData = data.item;
        }
        renderLists();
    } catch (e) {
        console.warn('[brain/app] failed to refresh expanded list:', e);
        showToast('Network error', 'error');
    }
}

async function addListItem(listId) {
    const input = document.getElementById(`addItemInput-${listId}`);
    if (!input) return;
    const content = input.value.trim();
    if (!content) return;

    try {
        const res = await apiFetch(`/lists/${listId}/items`, {
            method: 'POST',
            body: JSON.stringify({ items: [content] }),
        });
        const data = await res.json();
        if (res.ok) {
            if (data.added === 0) {
                showToast('Already on the list', 'info');
            } else {
                input.value = '';
            }
            await refreshExpandedList(listId);
        } else {
            showToast(data.error || 'Failed to add item', 'error');
        }
    } catch (e) {
        console.warn('[brain/app] failed to add list item:', e);
        showToast('Network error', 'error');
    }
}

async function removeListItem(listId, content) {
    try {
        const res = await apiFetch(`/lists/${listId}/items/batch`, {
            method: 'DELETE',
            body: JSON.stringify({ items: [content] }),
        });
        if (res.ok) {
            await refreshExpandedList(listId);
        } else {
            const err = await res.json().catch(() => ({}));
            showToast(err.error || 'Failed to remove item', 'error');
        }
    } catch (e) {
        console.warn('[brain/app] failed to remove list item:', e);
        showToast('Network error', 'error');
    }
}

async function toggleListItem(listId, content, checked) {
    const endpoint = checked ? 'check' : 'uncheck';
    try {
        const res = await apiFetch(`/lists/${listId}/items/${endpoint}`, {
            method: 'PUT',
            body: JSON.stringify({ items: [content] }),
        });
        if (res.ok) {
            await refreshExpandedList(listId);
        } else {
            const err = await res.json().catch(() => ({}));
            showToast(err.error || 'Failed to update item', 'error');
        }
    } catch (e) {
        console.warn('[brain/app] failed to toggle list item:', e);
        showToast('Network error', 'error');
    }
}

function openCreateList() {
    document.getElementById('createListName').value = '';
    document.getElementById('createListModal').classList.remove('hidden');
    document.getElementById('createListName').focus();
}

function openRenameList(id, name) {
    renamingListId = id;
    document.getElementById('renameListName').value = name;
    document.getElementById('renameListModal').classList.remove('hidden');
    document.getElementById('renameListName').focus();
}

function openDeleteList(id, name) {
    deletingListId = id;
    document.getElementById('deleteListDesc').textContent = `Delete "${name}"? This cannot be undone.`;
    document.getElementById('deleteListModal').classList.remove('hidden');
}

// ==========================================
// Lists Event Delegation
// ==========================================
document.getElementById('listsContainer').addEventListener('click', (e) => {
    // Rename button
    const renameBtn = e.target.closest('[data-action="rename"]');
    if (renameBtn) {
        openRenameList(renameBtn.dataset.listId, renameBtn.dataset.listName);
        return;
    }

    // Delete button
    const deleteBtn = e.target.closest('[data-action="delete"]');
    if (deleteBtn) {
        openDeleteList(deleteBtn.dataset.listId, deleteBtn.dataset.listName);
        return;
    }

    // Remove item button
    const removeBtn = e.target.closest('.list-item__remove');
    if (removeBtn) {
        removeListItem(removeBtn.dataset.listId, removeBtn.dataset.content);
        return;
    }

    // Add item button
    const addBtn = e.target.closest('.list-card__add-btn');
    if (addBtn) {
        addListItem(addBtn.dataset.listId);
        return;
    }

    // Toggle expand/collapse on header
    const header = e.target.closest('.list-card__header');
    if (header) {
        toggleListExpand(header.dataset.listId);
    }
});

document.getElementById('listsContainer').addEventListener('change', (e) => {
    if (e.target.matches('.list-item__checkbox input[type="checkbox"]')) {
        const item = e.target.closest('.list-item');
        if (item) {
            toggleListItem(item.dataset.listId, item.dataset.content, e.target.checked);
        }
    }
});

document.getElementById('listsContainer').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && e.target.classList.contains('list-card__add-input')) {
        e.preventDefault();
        addListItem(e.target.dataset.listId);
    }
});

// ==========================================
// Lists Modal Event Listeners
// ==========================================
document.getElementById('newListBtn').addEventListener('click', openCreateList);

document.getElementById('closeCreateListModal').addEventListener('click', () => {
    document.getElementById('createListModal').classList.add('hidden');
});
document.getElementById('cancelCreateListBtn').addEventListener('click', () => {
    document.getElementById('createListModal').classList.add('hidden');
});

document.getElementById('createListForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const name = document.getElementById('createListName').value.trim();
    if (!name) return;

    const btn = e.target.querySelector('button[type="submit"]');
    btn.disabled = true;
    btn.textContent = 'Creating…';

    try {
        const res = await apiFetch('/lists', {
            method: 'POST',
            body: JSON.stringify({ name }),
        });
        const data = await res.json();
        if (res.ok) {
            document.getElementById('createListModal').classList.add('hidden');
            showToast('List created', 'success');
            // Pre-set expanded state so new list opens automatically
            expandedListId = data.item.id;
            expandedListData = data.item;
            await loadLists();
        } else {
            showToast(data.error || 'Failed to create list', 'error');
        }
    } catch (e) {
        console.warn('[brain/app] failed to create list:', e);
        showToast('Network error', 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = 'Create';
    }
});

document.getElementById('closeRenameListModal').addEventListener('click', () => {
    document.getElementById('renameListModal').classList.add('hidden');
    renamingListId = null;
});
document.getElementById('cancelRenameListBtn').addEventListener('click', () => {
    document.getElementById('renameListModal').classList.add('hidden');
    renamingListId = null;
});

document.getElementById('renameListForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!renamingListId) return;
    const name = document.getElementById('renameListName').value.trim();
    if (!name) return;

    const btn = e.target.querySelector('button[type="submit"]');
    btn.disabled = true;
    btn.textContent = 'Renaming…';

    try {
        const res = await apiFetch(`/lists/${renamingListId}/rename`, {
            method: 'PUT',
            body: JSON.stringify({ name }),
        });
        const data = await res.json();
        if (res.ok) {
            document.getElementById('renameListModal').classList.add('hidden');
            showToast('List renamed', 'success');
            const prevExpanded = renamingListId;
            renamingListId = null;
            if (expandedListId === prevExpanded) {
                await refreshExpandedList(prevExpanded);
            } else {
                await loadLists();
            }
        } else {
            showToast(data.error || 'Failed to rename list', 'error');
        }
    } catch (e) {
        console.warn('[brain/app] failed to rename list:', e);
        showToast('Network error', 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = 'Rename';
    }
});

document.getElementById('cancelDeleteListBtn').addEventListener('click', () => {
    document.getElementById('deleteListModal').classList.add('hidden');
    deletingListId = null;
});

document.getElementById('confirmDeleteListBtn').addEventListener('click', async () => {
    if (!deletingListId) return;
    const id = deletingListId;
    document.getElementById('deleteListModal').classList.add('hidden');

    try {
        const res = await apiFetch(`/lists/${id}`, { method: 'DELETE' });
        if (res.ok) {
            showToast('List deleted', 'success');
            if (expandedListId === id) {
                expandedListId = null;
                expandedListData = null;
            }
            deletingListId = null;
            await loadLists();
        } else {
            const err = await res.json().catch(() => ({}));
            showToast(err.error || 'Failed to delete list', 'error');
            deletingListId = null;
        }
    } catch (e) {
        console.warn('[brain/app] failed to delete list:', e);
        showToast('Network error', 'error');
        deletingListId = null;
    }
});

// ==========================================
// Cognition — Observability Sub-tabs
// ==========================================

// Sub-tab switching
document.getElementById('cognitionSubtabs').addEventListener('click', (e) => {
    const btn = e.target.closest('.filter-tab');
    if (!btn) return;
    const subtab = btn.dataset.subtab;
    if (subtab === activeSubtab) return;

    activeSubtab = subtab;
    document.querySelectorAll('#cognitionSubtabs .filter-tab').forEach(t => t.classList.remove('active'));
    btn.classList.add('active');
    document.querySelectorAll('.cognition-subpanel').forEach(p => p.classList.remove('active'));
    document.getElementById(`subtab-${subtab}`).classList.add('active');

    loadCognitionSubtab(subtab);
});

// Refresh button
document.getElementById('obsRefreshBtn').addEventListener('click', () => {
    if (activeSubtab !== 'memory') {
        delete obsLoaded[activeSubtab];
        delete obsData[activeSubtab];
    }
    loadCognitionSubtab(activeSubtab);
});

function loadCognitionSubtab(subtab) {
    if (subtab === 'memory') { loadRecordsObs(); return; }
    if (subtab === 'personality') { loadPersonality(); return; }
    if (obsLoaded[subtab]) return; // Already cached

    const loaders = {
        tools: loadToolsObs,
        tasks: loadTasksObs,
        worldstate: loadWorldStateObs,
        errors: loadErrorsObs,
    };
    if (loaders[subtab]) loaders[subtab]();
}

// ── Shared helpers ──

function obsSkeletonBlock(height) {
    return `<div class="obs-skeleton" style="height:${height}px;margin-bottom:14px"></div>`;
}

function obsSetTimestamp(isoStr) {
    const el = document.getElementById('obsTimestamp');
    if (!el || !isoStr) return;
    try {
        const d = new Date(isoStr);
        el.textContent = 'Updated ' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } catch (e) { console.warn('[brain/app] obsSetTimestamp failed:', e); el.textContent = ''; }
}

// ── Memory Records ──

let activeRecordsSource = 'episodes';
let recordsQuery = '';
let recordsOffset = 0;
let recordsRows = [];
let _recordsDebounceTimer = null;

function _recordsSetStatus(msg) {
    const statusEl = document.getElementById('recordsStatus');
    const tableEl = document.getElementById('recordsTable');
    if (msg) {
        statusEl.textContent = msg;
        statusEl.style.display = '';
        tableEl.style.display = 'none';
    } else {
        statusEl.style.display = 'none';
        tableEl.style.display = '';
    }
}

function _recordsAppendRows(rows) {
    const tbody = document.getElementById('recordsTableBody');
    for (const r of rows) {
        const tr = document.createElement('tr');
        const createdText = r.created ? new Date(r.created).toLocaleDateString() : '—';
        const lastAccessedText = r.last_accessed ? timeAgo(r.last_accessed) : '—';
        const valueFull = r.value == null ? '' : String(r.value);
        const valueDisplay = valueFull.length > 500 ? valueFull.slice(0, 500) + '…' : valueFull;
        tr.innerHTML = `<td>${escapeHtml(createdText)}</td>` +
            `<td>${escapeHtml(lastAccessedText)}</td>` +
            `<td><code>${escapeHtml(String(r.key))}</code></td>` +
            `<td title="${escapeHtml(valueFull)}">${escapeHtml(valueDisplay)}</td>`;
        tbody.appendChild(tr);
    }
}

async function _recordsFetch(append) {
    if (!append) {
        recordsRows = [];
        recordsOffset = 0;
        document.getElementById('recordsTableBody').innerHTML = '';
    }

    if (!append) _recordsSetStatus('Loading…');

    const params = new URLSearchParams({ source: activeRecordsSource, offset: recordsOffset });
    if (recordsQuery) params.set('q', recordsQuery);

    try {
        const res = await apiFetch(`/system/observability/records?${params}`);
        if (!res.ok) throw new Error('Failed to load');
        const data = await res.json();
        obsSetTimestamp(data.generated_at);

        recordsRows = recordsRows.concat(data.rows || []);
        _recordsAppendRows(data.rows || []);

        const loadMoreBtn = document.getElementById('recordsLoadMore');
        if (data.has_more) {
            loadMoreBtn.classList.remove('hidden');
        } else {
            loadMoreBtn.classList.add('hidden');
        }

        if (recordsRows.length === 0) {
            _recordsSetStatus('No records found.');
        } else {
            _recordsSetStatus(null);
        }
    } catch (e) {
        console.warn('[brain/app] failed to load memory records:', e);
        _recordsSetStatus('Could not load records.');
        document.getElementById('recordsLoadMore').classList.add('hidden');
    }
}

function loadRecordsObs() {
    _recordsFetch(false);
}

document.getElementById('recordsSourceSwitcher').addEventListener('click', (e) => {
    const btn = e.target.closest('.filter-tab');
    if (!btn) return;
    const src = btn.dataset.source;
    if (src === activeRecordsSource) return;
    clearTimeout(_recordsDebounceTimer);
    activeRecordsSource = src;
    document.querySelectorAll('#recordsSourceSwitcher .filter-tab').forEach(t => t.classList.remove('active'));
    btn.classList.add('active');
    _recordsFetch(false);
});

document.getElementById('recordsSearchInput').addEventListener('input', (e) => {
    clearTimeout(_recordsDebounceTimer);
    _recordsDebounceTimer = setTimeout(() => {
        recordsQuery = e.target.value.trim();
        _recordsFetch(false);
    }, 300);
});

document.getElementById('recordsLoadMore').addEventListener('click', () => {
    recordsOffset += 250;
    _recordsFetch(true);
});

// ── Tools ──

async function loadToolsObs() {
    const el = document.getElementById('toolsObsContent');
    el.innerHTML = obsSkeletonBlock(40) + obsSkeletonBlock(120);
    try {
        const res = await apiFetch('/system/observability/tools');
        if (!res.ok) throw new Error('Failed to load');
        const data = await res.json();
        obsData.tools = data;
        obsLoaded.tools = true;
        obsSetTimestamp(data.generated_at);

        const tools = data.tools || [];
        if (tools.length === 0) {
            el.innerHTML = '<p class="obs-summary">No tools have been used yet.</p>';
            return;
        }
        let html = `<p class="obs-summary">${tools.length} tool${tools.length === 1 ? '' : 's'} used.</p>`;
        html += '<table class="obs-table"><thead><tr><th>Tool</th><th>Count</th><th>Last Used</th></tr></thead><tbody>';
        for (const t of tools) {
            const lastUsed = t.last_used_at ? timeAgo(t.last_used_at) : 'never';
            html += `<tr><td>${escapeHtml(t.tool_name)}</td><td>${t.count}</td><td>${escapeHtml(lastUsed)}</td></tr>`;
        }
        html += '</tbody></table>';
        el.innerHTML = html;
    } catch (e) {
        console.warn('[brain/app] failed to load tool data:', e);
        el.innerHTML = '<div class="obs-empty">Could not load tool data.</div>';
    }
}

// ── Working On (Tasks) ──

async function loadTasksObs() {
    const el = document.getElementById('tasksContent');
    el.innerHTML = obsSkeletonBlock(40) + obsSkeletonBlock(120);

    try {
        const res = await apiFetch('/system/observability/tasks');
        if (!res.ok) throw new Error('Failed to load');
        const data = await res.json();
        obsData.tasks = data;
        obsLoaded.tasks = true;
        obsSetTimestamp(data.generated_at);

        const tasks = data.persistent_tasks || [];

        let html = `<p class="obs-summary">Chalie is currently working on ${tasks.length} background task${tasks.length === 1 ? '' : 's'}.</p>`;

        // Persistent tasks
        if (tasks.length > 0) {
            html += '<div class="obs-section-title">Background Tasks</div>';
            const statusLabels = { accepted: 'Accepted', in_progress: 'In Progress', paused: 'Paused' };
            for (const t of tasks) {
                const label = statusLabels[t.status] || t.status;
                const badgeClass = t.status === 'paused' ? '--paused' : '--active';
                const progress = t.max_iterations ? `${t.iterations_used || 0} / ${t.max_iterations} iterations` : '';
                html += `<div class="obs-task-card">
                    <div class="obs-task-card__header">
                        <span class="obs-task-card__title">${escapeHtml(t.goal || 'Untitled task')}</span>
                        <span class="obs-task-card__badge ${badgeClass}">${escapeHtml(label)}</span>
                    </div>
                    <div class="obs-task-card__meta">
                        ${t.priority ? 'Priority: ' + escapeHtml(String(t.priority)) + ' · ' : ''}${progress}
                    </div>
                    <div class="obs-task-card__actions">
                        <button class="obs-task-btn --cancel" data-task-cancel="${t.id}">Cancel</button>
                    </div>
                </div>`;
            }
        } else {
            html += '<div class="obs-section-title">Background Tasks</div>';
            html += '<div class="obs-empty">No active background tasks.</div>';
        }

        el.innerHTML = html;

        // Wire cancel buttons
        el.querySelectorAll('[data-task-cancel]').forEach(btn => {
            btn.addEventListener('click', async (e) => {
                const taskId = e.target.dataset.taskCancel;
                if (!confirm('Cancel this task?')) return;
                try {
                    const r = await apiFetch(`/system/observability/tasks/${taskId}`, { method: 'DELETE' });
                    if (r.ok) loadTasksObs();
                    else alert('Failed to cancel task');
                } catch (e) { console.warn('[brain/app] task cancel failed:', e); alert('Failed to cancel task'); }
            });
        });
    } catch (e) {
        console.warn('[brain/app] failed to load task data:', e);
        el.innerHTML = '<div class="obs-empty">Could not load task data.</div>';
    }
}


async function loadWorldStateObs() {
    const el = document.getElementById('worldStateContent');
    el.innerHTML = obsSkeletonBlock(60) + obsSkeletonBlock(120);

    try {
        const res = await apiFetch('/system/observability/world-state');
        if (!res.ok) throw new Error('Failed to load');
        const data = await res.json();
        obsData.worldstate = data;
        obsLoaded.worldstate = true;

        const rendered = data.rendered || '';
        const inputs = data.inputs || {};

        let html = '';

        // ── Rendered block (literal text the LLM sees) ──
        html += '<div class="obs-section-title">Rendered (what the LLM sees)</div>';
        if (rendered) {
            html += `<pre class="obs-world-state-raw" style="font-family:monospace;white-space:pre-wrap">${escapeHtml(rendered)}</pre>`;
        } else {
            html += '<div class="obs-empty">World state is empty — nothing salient right now.</div>';
        }

        // ── Raw inputs table ──
        html += '<div class="obs-section-title" style="margin-top:20px">Raw Inputs</div>';
        const inputKeys = ['telemetry', 'signals', 'schedule', 'bg_processes'];
        html += '<table class="obs-inputs-table" style="width:100%;border-collapse:collapse;font-size:12px">';
        html += '<thead><tr><th style="text-align:left;padding:6px 8px;border-bottom:1px solid var(--border)">Source</th><th style="text-align:left;padding:6px 8px;border-bottom:1px solid var(--border)">Data</th></tr></thead><tbody>';
        for (const key of inputKeys) {
            const val = inputs[key];
            const isEmpty = val == null || (typeof val === 'object' && Object.keys(val).length === 0) || (Array.isArray(val) && val.length === 0);
            const display = isEmpty ? '<em style="opacity:0.4">empty</em>' : `<pre style="margin:0;white-space:pre-wrap;font-size:11px">${escapeHtml(JSON.stringify(val, null, 2))}</pre>`;
            html += `<tr><td style="padding:6px 8px;border-bottom:1px solid var(--border);vertical-align:top;white-space:nowrap">${key}</td><td style="padding:6px 8px;border-bottom:1px solid var(--border)">${display}</td></tr>`;
        }
        html += '</tbody></table>';

        el.innerHTML = html;
    } catch (e) {
        console.warn('[brain/app] failed to load world state data:', e);
        el.innerHTML = '<div class="obs-empty">Could not load world state data.</div>';
    }
}

// ── Error Log ──

async function loadErrorsObs() {
    const el = document.getElementById('subtab-errors');
    el.innerHTML = obsSkeletonBlock(40) + obsSkeletonBlock(120);
    try {
        const res = await apiFetch('/system/observability/errors');
        if (!res.ok) throw new Error('Failed to load');
        const data = await res.json();
        obsLoaded.errors = true;
        obsSetTimestamp(data.generated_at);

        const errors = data.errors || [];
        if (errors.length === 0) {
            el.innerHTML = '<div class="obs-empty">No errors logged.</div>';
            return;
        }
        const frag = document.createDocumentFragment();
        for (const entry of errors) {
            const row = document.createElement('div');
            row.className = 'error-log-row';
            const ts = document.createElement('time');
            ts.dateTime = entry.timestamp;
            ts.textContent = entry.timestamp;
            const msg = document.createElement('span');
            msg.textContent = entry.message;
            row.appendChild(ts);
            row.appendChild(msg);
            frag.appendChild(row);
        }
        el.innerHTML = '';
        el.appendChild(frag);
    } catch (e) {
        console.warn('[brain/app] failed to load error log:', e);
        el.innerHTML = '<div class="obs-empty">Could not load error log.</div>';
    }
}

// Relative time helper
function timeAgo(isoStr) {
    try {
        const d = new Date(isoStr);
        const now = Date.now();
        const diff = now - d.getTime();
        if (diff < 60000) return 'just now';
        if (diff < 3600000) return Math.floor(diff / 60000) + ' min ago';
        if (diff < 86400000) return Math.floor(diff / 3600000) + 'h ago';
        return Math.floor(diff / 86400000) + 'd ago';
    } catch (e) { console.warn('[brain/app] timeAgo failed:', e); return 'unknown'; }
}

// ==========================================
// Helpers
// ==========================================
function escapeHtml(str) {
    if (!str) return '';
    return String(str)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#039;');
}

const ACRONYMS = {
    cot: 'CoT', ai: 'AI', api: 'API', llm: 'LLM', url: 'URL', id: 'ID',
    ui: 'UI', ux: 'UX', nlp: 'NLP', ocr: 'OCR', sql: 'SQL', html: 'HTML',
    css: 'CSS', js: 'JS', json: 'JSON', xml: 'XML', pdf: 'PDF', mcp: 'MCP',
    os: 'OS', io: 'IO', tts: 'TTS', stt: 'STT', rag: 'RAG',
};

function humanizeSlug(str) {
    if (!str) return '';
    return String(str)
        .replaceAll(/[_-]+/g, ' ')
        .split(' ')
        .filter(Boolean)
        .map(w => ACRONYMS[w.toLowerCase()] || (w.charAt(0).toUpperCase() + w.slice(1).toLowerCase()))
        .join(' ');
}

// ==========================================
// Documents Tab
// ==========================================

let allDocuments = [];
let docFilter = 'active';
let docGroupBy = 'all'; // 'all', 'doc_category', 'doc_project', 'doc_date'

async function loadDocuments() {
    const el = document.getElementById('docList');
    el.innerHTML = '<div class="loading">Loading documents...</div>';
    try {
        const res = await apiFetch('/documents?include_deleted=true');
        if (!res.ok) throw new Error('Failed to load');
        const data = await res.json();
        allDocuments = data.items || [];
        renderDocuments();
        loadWatchedFolders();
    } catch (e) {
        console.warn('[brain/app] failed to load documents:', e);
        el.innerHTML = '<div class="empty-state"><p>Failed to load documents.</p></div>';
    }
}

function renderDocumentRow(doc) {
    const meta = doc.extracted_metadata || {};
    const docType = meta.document_type?.value || '';
    const typeBadge = docType && docType !== 'document'
        ? `<span class="doc-type-badge">${escapeHtml(docType)}</span>`
        : '';

    const categoryBadge = doc.doc_category
        ? `<span class="doc-category-badge">${escapeHtml(doc.doc_category)}</span>`
        : '';

    const size = doc.file_size_bytes
        ? (doc.file_size_bytes > 1024 * 1024
            ? `${(doc.file_size_bytes / 1024 / 1024).toFixed(1)} MB`
            : `${Math.round(doc.file_size_bytes / 1024)} KB`)
        : '';

    const pages = doc.page_count ? `${doc.page_count}p` : '';
    const chunks = `${doc.chunk_count || 0} chunks`;
    const date = doc.doc_date || (doc.created_at ? new Date(doc.created_at).toLocaleDateString() : '');

    let statusClass;
    if (doc.status === 'ready') statusClass = 'status-ready';
    else if (doc.status === 'failed') statusClass = 'status-error';
    else if (doc.status === 'processing') statusClass = 'status-building';
    else statusClass = '';

    const isDeleted = !!doc.deleted_at;

    let actions = '';
    if (isDeleted) {
        actions = `<button class="tool-card__btn doc-row__restore" title="Restore"><i data-lucide="undo-2"></i></button>
                   <button class="tool-card__btn doc-row__purge" title="Permanently delete"><i data-lucide="trash-2"></i></button>`;
    } else {
        actions = `<button class="tool-card__btn doc-row__preview" title="Preview"><i data-lucide="eye"></i></button>
                   <button class="tool-card__btn doc-row__delete" title="Delete"><i data-lucide="x"></i></button>`;
    }

    const imageThumbnail = doc.mime_type?.startsWith('image/')
        ? `<div class="doc-row__image-preview"><img src="${API_BASE}/documents/${doc.id}/preview" alt="${escapeHtml(doc.original_name)}" class="doc-row__image-thumb" loading="lazy"></div>`
        : '';

    return `<div class="doc-row ${isDeleted ? 'doc-row--deleted' : ''}" data-doc-id="${doc.id}">
        ${imageThumbnail}
        <div class="doc-row__info">
            <div class="doc-row__name">
                <span class="doc-icon">${getDocIcon(doc.mime_type)}</span>
                <span>${escapeHtml(doc.original_name)}</span>
                ${doc.watched_folder_id ? '<span class="doc-folder-badge" title="From watched folder">⊙</span>' : ''}
                ${categoryBadge}
                ${typeBadge}
                <span class="doc-status ${statusClass}">${doc.status}</span>
            </div>
            <div class="doc-row__meta">
                ${[size, pages, chunks, date].filter(Boolean).join(' · ')}
            </div>
        </div>
        <div class="doc-row__actions">${actions}</div>
    </div>`;
}

function renderDocuments() {
    const el = document.getElementById('docList');
    let docs = allDocuments;

    if (docFilter === 'active') {
        docs = docs.filter(d => !d.deleted_at && d.status === 'ready' && d.source_type !== 'chat_image');
    } else if (docFilter === 'processing') {
        docs = docs.filter(d => !d.deleted_at && ['pending', 'processing', 'awaiting_confirmation'].includes(d.status));
    } else if (docFilter === 'uploads') {
        docs = docs.filter(d => !d.deleted_at && d.source_type === 'chat_image');
    } else if (docFilter === 'deleted') {
        docs = docs.filter(d => d.deleted_at);
    }

    // Apply search filter
    const search = (document.getElementById('docSearchInput')?.value || '').trim().toLowerCase();
    if (search) {
        docs = docs.filter(d =>
            d.original_name.toLowerCase().includes(search)
            || (d.doc_category || '').toLowerCase().includes(search)
            || (d.doc_project || '').toLowerCase().includes(search)
        );
    }

    if (docs.length === 0) {
        el.innerHTML = '<div class="empty-state"><p>No documents found.</p></div>';
        return;
    }

    // Grouped view
    if (docGroupBy !== 'all') {
        const groups = {};
        for (const doc of docs) {
            let key;
            if (docGroupBy === 'doc_date') {
                key = doc.doc_date ? doc.doc_date.substring(0, 4) : 'Unknown';
            } else {
                key = doc[docGroupBy] || 'Uncategorized';
            }
            if (!groups[key]) groups[key] = [];
            groups[key].push(doc);
        }

        // Sort groups: Uncategorized/Unknown last, rest by count desc
        const sortedKeys = Object.keys(groups).sort((a, b) => {
            if (a === 'Uncategorized' || a === 'Unknown') return 1;
            if (b === 'Uncategorized' || b === 'Unknown') return -1;
            return groups[b].length - groups[a].length;
        });

        el.innerHTML = sortedKeys.map(key => `
            <div class="doc-group">
                <div class="doc-group__header">
                    <span class="doc-group__chevron">▾</span>
                    <span class="doc-group__name">${escapeHtml(key)}</span>
                    <span class="doc-group__count">${groups[key].length}</span>
                </div>
                <div class="doc-group__items">
                    ${groups[key].map(renderDocumentRow).join('')}
                </div>
            </div>
        `).join('');
    } else {
        // Flat view (default)
        el.innerHTML = docs.map(renderDocumentRow).join('');
    }

    if (typeof lucide !== 'undefined') lucide.createIcons({ node: el });
}

function getDocIcon(mime) {
    if (mime?.includes('pdf')) return '<i data-lucide="file-type"></i>';
    if (mime?.includes('word') || mime?.includes('docx')) return '<i data-lucide="file-text"></i>';
    if (mime?.includes('presentation') || mime?.includes('pptx')) return '<i data-lucide="presentation"></i>';
    if (mime?.includes('html')) return '<i data-lucide="file-code"></i>';
    if (mime?.includes('image')) return '<i data-lucide="image"></i>';
    return '<i data-lucide="file-text"></i>';
}

async function openMetaEditor(id) {
    const overlay = document.getElementById('docMetaOverlay');
    const titleEl = document.getElementById('docMetaTitle');
    const pillsEl = document.getElementById('docMetaPills');
    const bodyEl  = document.getElementById('docMetaBody');
    const footerEl = document.getElementById('docMetaFooter');

    titleEl.textContent = 'Loading...';
    pillsEl.innerHTML   = '';
    bodyEl.innerHTML    = '<div class="loading">Loading...</div>';
    footerEl.innerHTML  = '';
    overlay.classList.remove('hidden');

    try {
        const res = await apiFetch(`/documents/${id}`);
        if (!res.ok) throw new Error('Failed to load');
        const data = await res.json();
        const doc = data.item;

        titleEl.textContent = doc.original_name;

        const em = doc.extracted_metadata || {};
        let pillsHtml = '';
        if (em.document_type?.value) pillsHtml += `<span class="doc-type-badge">${escapeHtml(em.document_type.value)}</span> `;
        if (doc.language)   pillsHtml += `<span class="doc-meta-pill">${escapeHtml(doc.language)}</span> `;
        if (doc.page_count) pillsHtml += `<span class="doc-meta-pill">${doc.page_count} pages</span> `;
        pillsHtml += `<span class="doc-meta-pill">${doc.chunk_count || 0} chunks</span>`;
        if (doc.doc_project) pillsHtml += ` <span class="doc-meta-pill doc-project-pill">${escapeHtml(doc.doc_project)}</span>`;
        if (doc.doc_date)    pillsHtml += ` <span class="doc-meta-pill">${escapeHtml(doc.doc_date)}</span>`;
        pillsEl.innerHTML = pillsHtml;

        bodyEl.innerHTML = `
            <div class="doc-meta-overlay-fields">
                <label>Category
                    <input type="text" class="doc-classify-input" id="docClassCategory"
                           value="${escapeHtml(doc.doc_category || '')}" placeholder="e.g. Invoice, Receipt...">
                </label>
                <label>Project
                    <input type="text" class="doc-classify-input" id="docClassProject"
                           value="${escapeHtml(doc.doc_project || '')}" placeholder="e.g. Home Renovation...">
                </label>
                <label>Date
                    <input type="date" class="doc-classify-input" id="docClassDate"
                           value="${doc.doc_date || ''}">
                </label>
            </div>`;

        footerEl.innerHTML = `<button class="tool-card__btn doc-meta-save-btn" data-doc-id="${id}">Save</button>`;
    } catch (e) {
        console.error('[DocMeta]', e);
        bodyEl.innerHTML = '<div class="obs-empty">Failed to load document.</div>';
    }
}

async function openPreview(id) {
    const overlay   = document.getElementById('docPreviewOverlay');
    const titleEl   = document.getElementById('docPreviewTitle');
    const metaEl    = document.getElementById('docPreviewMeta');
    const bodyEl    = document.getElementById('docPreviewBody');
    const footerEl  = document.getElementById('docPreviewFooter');

    titleEl.textContent = 'Loading...';
    metaEl.innerHTML    = '';
    bodyEl.innerHTML    = '<div class="loading">Loading content...</div>';
    footerEl.innerHTML  = '';
    overlay.classList.remove('hidden');

    try {
        const res = await apiFetch(`/documents/${id}`);
        if (!res.ok) throw new Error('Failed to load');
        const data = await res.json();
        const doc = data.item;

        titleEl.textContent = doc.original_name;

        const em = doc.extracted_metadata || {};
        let pillsHtml = '';
        if (em.document_type?.value) pillsHtml += `<span class="doc-type-badge">${escapeHtml(em.document_type.value)}</span> `;
        if (doc.language)   pillsHtml += `<span class="doc-meta-pill">${escapeHtml(doc.language)}</span> `;
        if (doc.page_count) pillsHtml += `<span class="doc-meta-pill">${doc.page_count} pages</span> `;
        pillsHtml += `<span class="doc-meta-pill">${doc.chunk_count || 0} chunks</span>`;
        metaEl.innerHTML = pillsHtml;

        footerEl.innerHTML = `<a class="btn btn-secondary" href="${API_BASE}/documents/${id}/download" download>
            <i data-lucide="download"></i> Download
        </a>`;
        if (typeof lucide !== 'undefined') lucide.createIcons({ node: footerEl });

        const mime = doc.mime_type || '';
        const previewUrl = `${API_BASE}/documents/${id}/preview`;

        if (mime.startsWith('image/')) {
            bodyEl.innerHTML = `<div class="doc-preview-image"><img src="${previewUrl}" alt="${escapeHtml(doc.original_name)}" style="max-width:100%;border-radius:6px;"></div>`;

        } else if (mime.includes('pdf')) {
            bodyEl.innerHTML = `<iframe src="${previewUrl}" style="width:100%;height:70vh;border:none;border-radius:6px;" title="PDF Preview"></iframe>`;

        } else if (mime.includes('presentationml') || mime.includes('pptx')) {
            bodyEl.innerHTML = `<iframe src="${previewUrl}" style="width:100%;height:70vh;border:none;border-radius:6px;" title="Presentation Preview"></iframe>`;

        } else if (mime.includes('wordprocessingml') || mime.includes('docx')) {
            bodyEl.innerHTML = '<div class="loading">Rendering document...</div>';
            try {
                const dlRes = await apiFetch(`/documents/${id}/download`);
                const arrayBuffer = await dlRes.arrayBuffer();
                const result = await mammoth.convertToHtml({ arrayBuffer });
                bodyEl.innerHTML = `<div class="doc-preview-html" style="font-family:sans-serif;line-height:1.6;padding:8px;">${result.value}</div>`;
            } catch (e) {
                console.error('[DocPreview/DOCX]', e);
                bodyEl.innerHTML = `<iframe src="${previewUrl}" style="width:100%;height:70vh;border:none;border-radius:6px;" title="Document Preview"></iframe>`;
            }

        } else if (mime.includes('html')) {
            try {
                const txtRes = await apiFetch(`/documents/${id}/preview`);
                const html = await txtRes.text();
                const iframe = document.createElement('iframe');
                iframe.sandbox = 'allow-same-origin';
                iframe.style.cssText = 'width:100%;height:70vh;border:none;border-radius:6px;';
                iframe.title = 'HTML Preview';
                iframe.srcdoc = html;
                bodyEl.innerHTML = '';
                bodyEl.appendChild(iframe);
            } catch (err) {
                console.error('[DocPreview] HTML render failed:', err);
                bodyEl.innerHTML = '<div class="obs-empty">Failed to load HTML preview.</div>';
            }

        } else if (mime.includes('markdown') || doc.original_name?.endsWith('.md')) {
            try {
                const txtRes = await apiFetch(`/documents/${id}/preview`);
                const text = await txtRes.text();
                const html = (typeof marked !== 'undefined') ? marked.parse(text) : `<pre>${escapeHtml(text)}</pre>`;
                bodyEl.innerHTML = `<div class="doc-preview-html" style="line-height:1.6;padding:8px;">${html}</div>`;
            } catch (e) {
                console.error('[DocPreview/Markdown]', e);
                bodyEl.innerHTML = '<div class="obs-empty">Failed to load Markdown preview.</div>';
            }

        } else if (mime.includes('json')) {
            try {
                const txtRes = await apiFetch(`/documents/${id}/preview`);
                const text = await txtRes.text();
                let formatted = text;
                try { formatted = JSON.stringify(JSON.parse(text), null, 2); } catch (_parseErr) { /* keep raw */ }
                bodyEl.innerHTML = `<pre style="white-space:pre-wrap;word-break:break-all;font-size:12px;">${escapeHtml(formatted)}</pre>`;
            } catch (e) {
                console.error('[DocPreview/JSON]', e);
                bodyEl.innerHTML = '<div class="obs-empty">Failed to load JSON preview.</div>';
            }

        } else {
            try {
                const txtRes = await apiFetch(`/documents/${id}/preview`);
                const text = await txtRes.text();
                bodyEl.innerHTML = `<pre style="white-space:pre-wrap;word-break:break-all;font-size:12px;">${escapeHtml(text)}</pre>`;
            } catch (e) {
                console.error('[DocPreview/Text]', e);
                bodyEl.innerHTML = '<div class="obs-empty">Failed to load preview.</div>';
            }
        }
    } catch (e) {
        console.error('[DocPreview]', e);
        bodyEl.innerHTML = '<div class="obs-empty">Failed to load document.</div>';
    }
}

async function saveDocClassification(id) {
    const overlay = document.getElementById('docMetaOverlay');
    const category = overlay.querySelector('#docClassCategory')?.value?.trim() || null;
    const project = overlay.querySelector('#docClassProject')?.value?.trim() || null;
    const date = overlay.querySelector('#docClassDate')?.value || null;
    try {
        const res = await apiFetch(`/documents/${id}/classify`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ category, project, date }),
        });
        if (res.ok) {
            showToast('Classification saved', 'success');
            loadDocuments();
        } else {
            showToast('Failed to save classification', 'error');
        }
    } catch (e) {
        console.error('[saveDocClassification]', e);
        showToast('Failed to save classification', 'error');
    }
}

async function deleteDocument(id) {
    try {
        const res = await apiFetch(`/documents/${id}`, { method: 'DELETE' });
        if (res.ok) {
            loadDocuments();
            showToast('Document deleted', 'success', {
                action: 'Undo',
                onAction: () => restoreDocument(id),
                duration: 6000,
            });
        } else {
            showToast('Failed to delete document', 'error');
        }
    } catch (e) { console.error('[deleteDocument]', e); showToast('Failed to delete document', 'error'); }
}

async function restoreDocument(id) {
    try {
        const res = await apiFetch(`/documents/${id}/restore`, { method: 'POST' });
        if (res.ok) {
            showToast('Document restored', 'success');
            loadDocuments();
        } else {
            showToast('Failed to restore document', 'error');
        }
    } catch (e) { console.error('[restoreDocument]', e); showToast('Failed to restore document', 'error'); }
}

async function purgeDocument(id) {
    if (!confirm('Permanently delete this document? This cannot be undone.')) return;
    try {
        const res = await apiFetch(`/documents/${id}/purge`, { method: 'DELETE' });
        if (res.ok) {
            showToast('Document permanently deleted', 'success');
            loadDocuments();
        } else {
            showToast('Failed to purge document', 'error');
        }
    } catch (e) { console.error('[purgeDocument]', e); showToast('Failed to purge document', 'error'); }
}

// Document filter tabs
document.getElementById('docFilters')?.addEventListener('click', (e) => {
    const btn = e.target.closest('.filter-tab');
    if (!btn) return;
    document.querySelectorAll('#docFilters .filter-tab').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    docFilter = btn.dataset.filter;
    renderDocuments();
});

// Document group tabs
document.getElementById('docGroupTabs')?.addEventListener('click', (e) => {
    const btn = e.target.closest('.group-tab');
    if (!btn) return;
    document.querySelectorAll('#docGroupTabs .group-tab').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    docGroupBy = btn.dataset.group;
    renderDocuments();
});

// Document search
document.getElementById('docSearchInput')?.addEventListener('input', () => renderDocuments());

// Document upload from brain dashboard
document.getElementById('docUploadBtn')?.addEventListener('click', () => {
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = '.pdf,.docx,.pptx,.html,.htm,.txt,.md,.csv,.json,.xml,.jpg,.jpeg,.png,.webp,.gif';
    fileInput.addEventListener('change', async () => {
        const file = fileInput.files[0];
        if (!file) return;
        const formData = new FormData();
        formData.append('file', file);
        try {
            const res = await fetch(`${API_BASE}/documents/upload`, {
                method: 'POST',
                credentials: 'same-origin',
                body: formData,
            });
            if (res.ok) {
                showToast('Document uploaded — processing...', 'success');
                setTimeout(loadDocuments, 3000);
            } else {
                const err = await res.json().catch(() => ({}));
                showToast(err.error || 'Upload failed', 'error');
            }
        } catch (e) { console.warn('[brain/app] document upload failed:', e); showToast('Upload failed', 'error'); }
    });
    fileInput.click();
});

// Document preview modal close
document.getElementById('docPreviewClose')?.addEventListener('click', () => {
    document.getElementById('docPreviewOverlay').classList.add('hidden');
});
document.getElementById('docPreviewOverlay')?.addEventListener('click', (e) => {
    if (e.target.id === 'docPreviewOverlay') e.target.classList.add('hidden');
});

// Document metadata editor modal
document.getElementById('docMetaClose')?.addEventListener('click', () => {
    document.getElementById('docMetaOverlay').classList.add('hidden');
});
document.getElementById('docMetaOverlay')?.addEventListener('click', (e) => {
    if (e.target.id === 'docMetaOverlay') {
        e.target.classList.add('hidden');
        return;
    }
    const saveBtn = e.target.closest('.doc-meta-save-btn');
    if (saveBtn) {
        const docId = saveBtn.dataset.docId;
        if (docId) saveDocClassification(docId);
    }
});

// Document list — delegated event handler
document.getElementById('docList')?.addEventListener('click', (e) => {
    const groupHeader = e.target.closest('.doc-group__header');
    if (groupHeader) {
        groupHeader.parentElement.classList.toggle('collapsed');
        return;
    }

    const row = e.target.closest('.doc-row[data-doc-id]');
    if (!row) return;
    const id = row.dataset.docId;

    if (e.target.closest('.doc-row__preview')) { openPreview(id);      return; }
    if (e.target.closest('.doc-row__delete'))  { deleteDocument(id);   return; }
    if (e.target.closest('.doc-row__restore')) { restoreDocument(id);  return; }
    if (e.target.closest('.doc-row__purge'))   { purgeDocument(id);    return; }
    if (e.target.closest('.doc-row__actions')) return;
    if (e.target.closest('.doc-row__info'))    { openMetaEditor(id);   return; }
});

// Global Escape key handler — closes topmost visible modal
document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    const previewOverlay    = document.getElementById('docPreviewOverlay');
    const metaOverlay       = document.getElementById('docMetaOverlay');
    const watchFolderModal  = document.getElementById('watchFolderModal');

    if (previewOverlay && !previewOverlay.classList.contains('hidden')) {
        previewOverlay.classList.add('hidden');
    } else if (metaOverlay && !metaOverlay.classList.contains('hidden')) {
        metaOverlay.classList.add('hidden');
    } else if (watchFolderModal && !watchFolderModal.classList.contains('hidden')) {
        watchFolderModal.classList.add('hidden');
    }
});

// ==========================================
// Watched Folders
// ==========================================

async function loadWatchedFolders() {
    try {
        const res = await apiFetch('/documents/watched-folders');
        if (!res.ok) return;
        const data = await res.json();
        const folders = data.items || [];
        const panel = document.getElementById('watchedFoldersPanel');
        if (folders.length > 0) {
            panel.classList.remove('hidden');
            renderWatchedFolders(folders);
        } else {
            panel.classList.add('hidden');
        }
    } catch (e) { console.error('[WatchedFolders] load error:', e); }
}


function renderWatchedFolders(folders) {
    const el = document.getElementById('watchedFoldersList');
    el.innerHTML = folders.map(f => {
        const disabled = !f.enabled;
        const lastScan = f.last_scan_at
            ? `Last scan: ${new Date(f.last_scan_at).toLocaleString()} · ${f.last_scan_files} files`
            : 'Not scanned yet';
        const error = f.last_scan_error
            ? `<span style="color:#ff4d4d;font-size:11px"> · ${escapeHtml(f.last_scan_error)}</span>`
            : '';

        return `<div class="watched-folder-row ${disabled ? 'watched-folder-disabled' : ''}">
            <div class="watched-folder-info">
                <div class="watched-folder-label">${escapeHtml(f.label || f.folder_path)}</div>
                <div class="watched-folder-path">${escapeHtml(f.folder_path)}</div>
                <div class="watched-folder-stats">${lastScan}${error}</div>
            </div>
            <div class="watched-folder-actions">
                <label class="toggle-switch" title="${disabled ? 'Enable' : 'Disable'}">
                    <input type="checkbox" class="wf-toggle" data-folder-id="${f.id}" ${disabled ? '' : 'checked'}>
                    <span class="slider"></span>
                </label>
                <button class="btn-icon scan-btn wf-scan" data-folder-id="${f.id}" title="Scan now">⟳</button>
                <button class="btn-icon wf-edit" data-folder-id="${f.id}" title="Edit">✎</button>
                <button class="btn-icon delete-btn wf-delete" data-folder-id="${f.id}" title="Delete">✕</button>
            </div>
        </div>`;
    }).join('');
}

// Watched folder event delegation
document.getElementById('watchedFoldersList')?.addEventListener('change', (e) => {
    const toggle = e.target.closest('.wf-toggle');
    if (toggle) toggleWatchFolder(toggle.dataset.folderId, toggle.checked);
});
document.getElementById('watchedFoldersList')?.addEventListener('click', (e) => {
    const scan = e.target.closest('.wf-scan');
    const edit = e.target.closest('.wf-edit');
    const del  = e.target.closest('.wf-delete');
    if (scan) { triggerFolderScan(scan.dataset.folderId); return; }
    if (edit) { openWatchFolderModal(edit.dataset.folderId); return; }
    if (del)  { deleteWatchFolder(del.dataset.folderId); return; }
});

// Toggle watched folders panel
document.getElementById('watchedFoldersToggle')?.addEventListener('click', () => {
    document.getElementById('watchedFoldersPanel').classList.toggle('collapsed');
});

// Watch Folder button
document.getElementById('docWatchFolderBtn')?.addEventListener('click', () => openWatchFolderModal());

let _currentBrowsePath = null;

async function openWatchFolderModal(editId) {
    const modal = document.getElementById('watchFolderModal');
    const title = document.getElementById('watchFolderModalTitle');
    const pathInput = document.getElementById('watchFolderPath');
    const editIdInput = document.getElementById('watchFolderEditId');
    const saveBtn = document.getElementById('saveWatchFolderBtn');

    editIdInput.value = editId || '';

    if (editId) {
        title.textContent = 'Edit Watched Folder';
        saveBtn.textContent = 'Save';
        try {
            const res = await apiFetch(`/documents/watched-folders`);
            const data = await res.json();
            const folder = (data.items || []).find(f => f.id === editId);
            if (folder) {
                pathInput.value = folder.folder_path;
                document.getElementById('watchFolderLabel').value = folder.label || '';
                const pats = Array.isArray(folder.file_patterns) ? folder.file_patterns : ['*'];
                document.getElementById('watchFolderPatterns').value = pats.join(', ');
                const igns = Array.isArray(folder.ignore_patterns) ? folder.ignore_patterns : [];
                document.getElementById('watchFolderIgnore').value = igns.join(', ');
                document.getElementById('watchFolderRecursive').checked = !!folder.recursive;
                document.getElementById('watchFolderInterval').value = folder.scan_interval || 300;
                _currentBrowsePath = folder.folder_path;
            }
        } catch (e) { console.warn('[brain/app] failed to load watched folder for edit:', e); }
    } else {
        title.textContent = 'Watch Folder';
        saveBtn.textContent = 'Watch';
        pathInput.value = '';
        document.getElementById('watchFolderLabel').value = '';
        document.getElementById('watchFolderPatterns').value = '*';
        document.getElementById('watchFolderIgnore').value = '.git, node_modules, __pycache__, build, dist, .DS_Store, Thumbs.db';
        document.getElementById('watchFolderRecursive').checked = true;
        document.getElementById('watchFolderInterval').value = 300;
        _currentBrowsePath = null;
    }

    modal.classList.remove('hidden');
    browseDirectory(_currentBrowsePath);
}

async function browseDirectory(path) {
    const listEl = document.getElementById('dirBrowserList');
    const pathEl = document.getElementById('dirBrowserPath');
    listEl.innerHTML = '<div class="loading">Loading…</div>';

    try {
        const body = path ? { path } : {};
        const res = await apiFetch('/documents/watched-folders/browse', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            listEl.innerHTML = `<div class="dir-browser-empty">${escapeHtml(err.error || 'Cannot read directory')}</div>`;
            return;
        }
        const data = await res.json();
        _currentBrowsePath = data.current;
        pathEl.textContent = data.current;

        // Update the hidden path field and derive label
        document.getElementById('watchFolderPath').value = data.current;
        if (!document.getElementById('watchFolderEditId').value) {
            const basename = data.current.split('/').filter(Boolean).pop() || '';
            if (!document.getElementById('watchFolderLabel').value) {
                document.getElementById('watchFolderLabel').value = basename;
            }
        }

        // Set parent nav
        const upBtn = document.getElementById('dirBrowserUp');
        upBtn.onclick = data.parent ? () => browseDirectory(data.parent) : null;
        upBtn.style.opacity = data.parent ? '1' : '0.3';

        if (data.directories.length === 0) {
            listEl.innerHTML = '<div class="dir-browser-empty">No subdirectories</div>';
        } else {
            listEl.innerHTML = data.directories.map(d =>
                `<div class="dir-browser-item" data-dir-path="${escapeHtml(data.current)}/${escapeHtml(d)}">${escapeHtml(d)}</div>`
            ).join('');
        }
    } catch (e) {
        console.warn('[brain/app] failed to browse directory:', e);
        listEl.innerHTML = '<div class="dir-browser-empty">Failed to browse directory</div>';
    }
}

// Directory browser — delegated click handler
document.getElementById('dirBrowserList')?.addEventListener('click', (e) => {
    const item = e.target.closest('.dir-browser-item');
    if (!item) return;
    browseDirectory(item.dataset.dirPath);
});

// Watch Folder form submit
document.getElementById('watchFolderForm')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const editId = document.getElementById('watchFolderEditId').value;
    const folderPath = document.getElementById('watchFolderPath').value;
    if (!folderPath) {
        showToast('Select a folder first', 'error');
        return;
    }

    const patternsRaw = document.getElementById('watchFolderPatterns').value;
    const ignoreRaw = document.getElementById('watchFolderIgnore').value;
    const filePatterns = patternsRaw.split(',').map(s => s.trim()).filter(Boolean);
    const ignorePatterns = ignoreRaw.split(',').map(s => s.trim()).filter(Boolean);

    const payload = {
        folder_path: folderPath,
        label: document.getElementById('watchFolderLabel').value.trim() || null,
        file_patterns: filePatterns,
        ignore_patterns: ignorePatterns,
        recursive: document.getElementById('watchFolderRecursive').checked,
        scan_interval: Number.parseInt(document.getElementById('watchFolderInterval').value, 10) || 300,
    };

    try {
        const url = editId
            ? `/documents/watched-folders/${editId}`
            : '/documents/watched-folders';
        const method = editId ? 'PUT' : 'POST';
        const res = await apiFetch(url, {
            method,
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        if (res.ok) {
            showToast(editId ? 'Watched folder updated' : 'Folder watch started', 'success');
            document.getElementById('watchFolderModal').classList.add('hidden');
            loadDocuments();
        } else {
            const err = await res.json().catch(() => ({}));
            showToast(err.error || 'Failed to save', 'error');
        }
    } catch (e) { console.warn('[brain/app] failed to save watched folder:', e); showToast('Failed to save watched folder', 'error'); }
});

// Close / cancel watch folder modal
document.getElementById('closeWatchFolderModal')?.addEventListener('click', () => {
    document.getElementById('watchFolderModal').classList.add('hidden');
});
document.getElementById('cancelWatchFolderBtn')?.addEventListener('click', () => {
    document.getElementById('watchFolderModal').classList.add('hidden');
});
document.getElementById('watchFolderModal')?.addEventListener('click', (e) => {
    if (e.target.id === 'watchFolderModal') e.target.classList.add('hidden');
});

async function toggleWatchFolder(id, enabled) {
    try {
        await apiFetch(`/documents/watched-folders/${id}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ enabled }),
        });
        loadWatchedFolders();
    } catch (e) { console.warn('[brain/app] failed to toggle folder:', e); showToast('Failed to toggle folder', 'error'); }
}

async function triggerFolderScan(id) {
    try {
        const res = await apiFetch(`/documents/watched-folders/${id}/scan`, { method: 'POST' });
        if (res.ok) {
            showToast('Scan requested — results in next cycle', 'success');
        } else {
            showToast('Failed to trigger scan', 'error');
        }
    } catch (e) { console.warn('[brain/app] failed to trigger scan:', e); showToast('Failed to trigger scan', 'error'); }
}

async function deleteWatchFolder(id) {
    if (!confirm('Remove this folder watch? Documents already indexed will remain.')) return;
    try {
        const res = await apiFetch(`/documents/watched-folders/${id}`, { method: 'DELETE' });
        if (res.ok) {
            showToast('Folder watch removed', 'success');
            loadDocuments();
        } else {
            showToast('Failed to remove folder watch', 'error');
        }
    } catch (e) { console.warn('[brain/app] failed to remove folder watch:', e); showToast('Failed to remove folder watch', 'error'); }
}

// ==========================================
// Capabilities Tab
// ==========================================

const _SELF_HOSTED_PROVIDERS = new Set(['nextcloud', 'synology', 'radicale']);
const _APP_PASSWORD_HINTS = {
    google: 'Google Account \u2192 Security \u2192 App Passwords (requires 2FA)',
    apple: 'Apple ID \u2192 Sign-In and Security \u2192 App-Specific Passwords',
    yahoo: 'Yahoo Account \u2192 Account Security \u2192 App Passwords',
    outlook: 'Microsoft Account \u2192 Security \u2192 App Passwords (requires 2FA)',
};

let capabilitiesData = [];

async function loadCapabilities() {
    const el = document.getElementById('capabilitiesList');
    el.innerHTML = '<div class="loading">Loading capabilities...</div>';
    try {
        const res = await apiFetch('/api/capabilities');
        if (res.status === 401) { window.location.replace('/login/?next=/brain/'); return; }
        if (!res.ok) throw new Error('Failed to load');
        const data = await res.json();
        capabilitiesData = data.capabilities || [];
        renderCapabilities();
    } catch (e) {
        console.warn('[brain/app] failed to load capabilities:', e);
        el.innerHTML = '<div class="empty-state"><p>Failed to load capabilities.</p></div>';
    }
}

function renderCapabilities() {
    const el = document.getElementById('capabilitiesList');
    if (!capabilitiesData.length) {
        el.innerHTML = '<div class="empty-state"><h3>No capabilities found</h3><p>No capability plugins are installed.</p></div>';
        return;
    }
    el.innerHTML = capabilitiesData.map(cap => {
        const connected = cap.connected;
        const statusClass = connected ? 'cap-status-connected' : 'cap-status-disconnected';
        const statusLabel = connected ? 'Connected' : 'Not connected';
        const syncInfo = cap.last_sync_at ? `Last sync: ${timeAgo(cap.last_sync_at)}` : 'Never synced';
        const providerList = (cap.providers || []).map(p => escapeHtml(p)).join(', ');

        return `
        <div class="cap-card" data-id="${escapeHtml(cap.id)}">
            <div class="cap-info">
                <div class="cap-name-row">
                    <span class="cap-status-dot ${statusClass}"></span>
                    <span class="cap-name">${escapeHtml(cap.name)}</span>
                    <span class="cap-version">${escapeHtml(cap.version)}</span>
                </div>
                <div class="cap-meta">
                    <span class="cap-status-label">${statusLabel}</span>
                    ${connected ? ` &middot; <span class="cap-sync">${syncInfo}</span>` : ''}
                    ${providerList ? ` &middot; ${providerList}` : ''}
                </div>
            </div>
            <div class="cap-actions">
                ${connected
                    ? `<button class="btn btn-danger btn-sm" data-cap-disconnect="${escapeHtml(cap.id)}">Disconnect</button>`
                    : `<button class="btn btn-primary btn-sm" data-cap-connect="${escapeHtml(cap.id)}">Connect</button>`
                }
            </div>
        </div>`;
    }).join('');

    el.querySelectorAll('[data-cap-connect]').forEach(btn => {
        btn.addEventListener('click', () => openCapSetup(btn.dataset.capConnect));
    });
    el.querySelectorAll('[data-cap-disconnect]').forEach(btn => {
        btn.addEventListener('click', () => disconnectCapability(btn.dataset.capDisconnect));
    });
}

function openCapSetup(capId) {
    document.getElementById('capSetupForm').reset();
    document.getElementById('capSetupId').value = capId;
    document.getElementById('capServerUrl').value = '';
    document.getElementById('capServerUrlGroup').classList.add('hidden');
    document.getElementById('capPasswordHint').textContent = '';

    const cap = capabilitiesData.find(c => c.id === capId);
    document.getElementById('capSetupTitle').textContent = `Connect ${cap ? cap.name : capId}`;

    // Populate provider options from capability's providers list
    const select = document.getElementById('capProvider');
    select.innerHTML = '<option value="">Select provider...</option>';
    if (cap && cap.providers) {
        for (const p of cap.providers) {
            const opt = document.createElement('option');
            opt.value = p;
            opt.textContent = p.charAt(0).toUpperCase() + p.slice(1);
            select.appendChild(opt);
        }
    }

    document.getElementById('capSetupOverlay').classList.remove('hidden');
    select.focus();
}

// Show/hide server URL based on provider selection
document.getElementById('capProvider').addEventListener('change', (e) => {
    const provider = e.target.value;
    const serverGroup = document.getElementById('capServerUrlGroup');
    const hint = document.getElementById('capPasswordHint');

    if (_SELF_HOSTED_PROVIDERS.has(provider)) {
        serverGroup.classList.remove('hidden');
    } else {
        serverGroup.classList.add('hidden');
        document.getElementById('capServerUrl').value = '';
    }

    hint.textContent = _APP_PASSWORD_HINTS[provider] || '';
});

// Close modal
document.getElementById('capSetupClose').addEventListener('click', () => {
    document.getElementById('capSetupOverlay').classList.add('hidden');
});
document.getElementById('capSetupCancel').addEventListener('click', () => {
    document.getElementById('capSetupOverlay').classList.add('hidden');
});

// Submit setup form
document.getElementById('capSetupForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const capId = document.getElementById('capSetupId').value;
    const provider = document.getElementById('capProvider').value;
    const username = document.getElementById('capUsername').value.trim();
    const password = document.getElementById('capPassword').value;
    const serverUrl = document.getElementById('capServerUrl').value.trim();

    if (!provider) { showToast('Select a provider', 'error'); return; }
    if (!username) { showToast('Username is required', 'error'); return; }
    if (!password) { showToast('App password is required', 'error'); return; }
    if (_SELF_HOSTED_PROVIDERS.has(provider) && !serverUrl) {
        showToast('Server URL is required for self-hosted providers', 'error');
        return;
    }

    const btn = document.getElementById('capSetupSubmit');
    btn.disabled = true;
    btn.textContent = 'Connecting...';

    const body = { provider, username, password };
    if (serverUrl) body.server_url = serverUrl;

    try {
        const res = await apiFetch(`/api/capabilities/${capId}/setup`, {
            method: 'POST',
            body: JSON.stringify(body),
        });
        if (res.status === 401) { window.location.replace('/login/?next=/brain/'); return; }
        const data = await res.json();
        if (res.ok) {
            document.getElementById('capSetupOverlay').classList.add('hidden');
            showToast('Connected successfully', 'success');
            await loadCapabilities();
        } else {
            showToast(data.error || 'Connection failed', 'error');
        }
    } catch (err) {
        console.warn('[brain/app] capability setup failed:', err);
        showToast('Network error', 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = 'Connect';
    }
});

async function disconnectCapability(capId) {
    if (!confirm(`Disconnect this capability? Stored credentials will be removed.`)) return;
    try {
        const res = await apiFetch(`/api/capabilities/${capId}/disconnect`, { method: 'POST' });
        if (res.ok) {
            showToast('Disconnected', 'success');
            await loadCapabilities();
        } else {
            const data = await res.json();
            showToast(data.error || 'Failed to disconnect', 'error');
        }
    } catch (err) {
        console.warn('[brain/app] capability disconnect failed:', err);
        showToast('Network error', 'error');
    }
}

// ==========================================
// Personality
// ==========================================

const PERSONALITY_SLIDERS = ['warmth', 'mood', 'expressiveness', 'curiosity', 'humor'];

let _personalityDebounceTimer = null;

function _getPersonalitySliderValues() {
    return PERSONALITY_SLIDERS.map(name => {
        const el = document.querySelector(`[data-slider="${name}"]`);
        return el ? Number.parseInt(el.value, 10) : 0;
    });
}

function _renderPersonalityVoice(voice) {
    const preview = document.getElementById('personalityVoicePreview');
    if (!preview) return;
    preview.textContent = voice || '';
}

function _applyPersonalityTuple(tup) {
    PERSONALITY_SLIDERS.forEach((name, i) => {
        const el = document.querySelector(`[data-slider="${name}"]`);
        if (el) el.value = tup[i] ?? 0;
    });
}

async function loadPersonality() {
    try {
        const res = await apiFetch('/settings/personality');
        if (!res.ok) {
            _renderPersonalityVoice('Failed to load personality settings.');
            return;
        }
        const data = await res.json();
        _applyPersonalityTuple(data.tuple);
        _renderPersonalityVoice(data.voice);
    } catch (err) {
        console.error('[personality] load failed', err);
        _renderPersonalityVoice('Network error loading personality settings.');
    }
}

async function _savePersonality(tup) {
    try {
        const res = await apiFetch('/settings/personality', {
            method: 'PUT',
            body: JSON.stringify({ tuple: tup }),
        });
        const data = await res.json();
        if (res.ok) {
            _renderPersonalityVoice(data.voice);
            return true;
        }
        showToast(data.error || 'Failed to save personality', 'error');
        return false;
    } catch (err) {
        console.error('[personality] save failed', err);
        showToast('Network error saving personality', 'error');
        return false;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    // Save button
    const saveBtn = document.getElementById('savePersonalityBtn');
    if (saveBtn) {
        saveBtn.addEventListener('click', async () => {
            if (await _savePersonality(_getPersonalitySliderValues())) {
                showToast('Personality saved', 'success');
            }
        });
    }

    // Live preview on slider drag (debounced 250 ms)
    document.querySelectorAll('.personality-range').forEach(slider => {
        slider.addEventListener('input', () => {
            clearTimeout(_personalityDebounceTimer);
            _personalityDebounceTimer = setTimeout(() => {
                _savePersonality(_getPersonalitySliderValues());
            }, 250);
        });
    });
});

// ==========================================
// Start
// ==========================================
init();

if (typeof lucide !== 'undefined') lucide.createIcons();
