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
let assignments = {};
let editPlatform = 'ollama';
let editingProviderId = null;
let deletingProviderId = null;

// Embodiment state
let embodyTools = [];           // full tool list from API
let settingsTool = null;        // tool name open in settings modal
let installMethod = 'git';      // 'git' or 'zip'
let pollTimer = null;           // setInterval id for build polling

// ==========================================
// LLM Jobs
// ==========================================
const JOBS = [
    { id: 'frontal-cortex', name: 'Frontal Cortex', desc: 'Core reasoning engine; orchestrates all response modes.', badge: '≥ 30B', badgeClass: 'badge-30b' },
    { id: 'frontal-cortex-respond', name: 'Respond Mode', desc: 'Primary voice of Chalie in normal conversation.', badge: '≥ 30B', badgeClass: 'badge-30b' },
    { id: 'frontal-cortex-act', name: 'Act Mode', desc: 'Plans and executes multi-step tool actions. Requires strong reasoning.', badge: '≥ 30B', badgeClass: 'badge-30b' },
    { id: 'frontal-cortex-clarify', name: 'Clarify Mode', desc: 'Asks clarifying questions when intent is ambiguous.', badge: '≥ 14B', badgeClass: 'badge-14b' },
    { id: 'frontal-cortex-proactive', name: 'Proactive Mode', desc: 'Translates spontaneous thoughts into outreach messages.', badge: '≥ 14B', badgeClass: 'badge-14b' },
    { id: 'mode-reflection', name: 'Mode Reflection', desc: 'Peer-reviews routing decisions during idle time.', badge: '≥ 14B', badgeClass: 'badge-14b' },
    { id: 'frontal-cortex-acknowledge', name: 'Acknowledge Mode', desc: 'Brief acknowledgments for greetings and simple inputs.', badge: '8B+', badgeClass: 'badge-8b' },
    { id: 'cognitive-triage', name: 'Cognitive Triage', desc: 'Routes user input to optimal cognitive branch (RESPOND/CLARIFY/ACT). Lightweight model preferred.', badge: '8B sufficient', badgeClass: 'badge-8b' },
    { id: 'memory-chunker', name: 'Memory Chunker', desc: 'Extracts gists, facts, and traits from exchanges. Runs async.', badge: '8B sufficient', badgeClass: 'badge-8b' },
    { id: 'episodic-memory', name: 'Episodic Memory', desc: 'Synthesises sessions into episodic narratives for long-term recall.', badge: '8B sufficient', badgeClass: 'badge-8b' },
    { id: 'semantic-memory', name: 'Semantic Memory', desc: 'Extracts concepts and relationships to build the knowledge graph.', badge: '8B sufficient', badgeClass: 'badge-8b' },
    { id: 'autobiography', name: 'Autobiography Synthesis', desc: 'Generates personal narrative summaries from stored memories (6h cycle).', badge: '8B sufficient', badgeClass: 'badge-8b' },
    { id: 'experience-assimilation', name: 'Experience Assimilation', desc: 'Evaluates tool outputs for novel knowledge worth storing.', badge: '8B sufficient', badgeClass: 'badge-8b' },
    { id: 'cognitive-drift', name: 'Cognitive Drift (DMN)', desc: 'Generates spontaneous thoughts during idle (Default Mode Network).', badge: '4B sufficient', badgeClass: 'badge-4b' },
    { id: 'mode-tiebreaker', name: 'Mode Tiebreaker', desc: 'Resolves ambiguous routing with binary A-vs-B decision. Must be fast.', badge: '4B sufficient', badgeClass: 'badge-4b' },
];

// ==========================================
// Platform Config
// ==========================================
const PLATFORM_CONFIG = {
    ollama: {
        desc: 'Run locally — no API key needed. Download from <a href="https://ollama.ai" target="_blank">ollama.ai</a>',
        hasHost: true,
        hasApiKey: false,
        modelPlaceholder: 'e.g. qwen3:8b',
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
        modelPlaceholder: 'e.g. gemini-2.0-flash',
        models: ['gemini-2.0-flash', 'gemini-2.0-flash-lite', 'gemini-2.0-pro'],
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
function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transition = 'opacity 0.3s';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// ==========================================
// Init
// ==========================================
async function init() {
    await resolveApiKey();
}

async function resolveApiKey() {
    try {
        const statusUrl = API_BASE ? `${API_BASE.replace(/\/$/, '')}/auth/status` : '/auth/status';
        const res = await fetch(statusUrl, { credentials: 'same-origin' });
        const data = res.ok ? await res.json() : {};

        // Only redirect to on-boarding for a completely fresh install (no account yet)
        if (!data.has_master_account) {
            window.location.replace('/on-boarding/');
            return;
        }
        // No session — show the dashboard login modal
        if (!data.has_session) {
            showLoginModal();
            return;
        }
        // Logged in — load dashboard regardless of provider state
        await loadData();
    } catch (err) {
        showToast('Cannot connect to backend. Is the API running?', 'error');
    }
}

function showLoginModal() {
    document.getElementById('loginModal').classList.remove('hidden');
}

document.getElementById('loginForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const username = document.getElementById('loginUsername').value.trim();
    const password = document.getElementById('loginPassword').value.trim();
    if (!username || !password) {
        showToast('Username and password required', 'error');
        return;
    }
    const btn = e.target.querySelector('button[type="submit"]');
    btn.disabled = true;
    btn.textContent = 'Logging in...';
    try {
        const res = await fetch('/auth/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'same-origin',
            body: JSON.stringify({ username, password }),
        });
        if (res.ok) {
            // Reload the page so resolveApiKey() runs fresh with the session cookie committed
            window.location.replace('/brain/');
        } else if (res.status === 401) {
            showToast('Invalid credentials', 'error');
        } else {
            const err = await res.json().catch(() => ({}));
            showToast(err.error || 'Login failed', 'error');
        }
    } catch {
        showToast('Network error', 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = 'Login';
    }
});

async function loadData() {
    try {
        const res = await apiFetch('/providers');
        if (res.ok) {
            const data = await res.json();
            providers = data.providers || [];
        } else if (res.status === 401) {
            // Session expired — show login modal
            showLoginModal();
            return;
        } else {
            showToast('Failed to load providers', 'error');
        }
    } catch (err) {
        showToast('Failed to connect to backend', 'error');
        return;
    }

    await loadAssignments();
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

    // Show/hide api key field
    document.getElementById('editApiKeyGroup').style.display = config.hasApiKey ? '' : 'none';

    // Update model input
    const modelInput = document.getElementById('editModel');
    modelInput.placeholder = config.modelPlaceholder;

    // Update datalist for curated platforms
    const datalist = document.getElementById('editModelSuggestions');
    datalist.innerHTML = '';
    if (config.models.length > 0 && platform !== 'ollama') {
        config.models.forEach(m => {
            const opt = document.createElement('option');
            opt.value = m;
            datalist.appendChild(opt);
        });
    }

    // For Anthropic: fetch models when api key is entered
    if (platform === 'anthropic') {
        const apiKeyInput = document.getElementById('editApiKey');
        apiKeyInput.oninput = debounce(() => {
            if (apiKeyInput.value.length > 20) {
                fetchAnthropicModels(apiKeyInput.value, 'editModelSuggestions');
            }
        }, 500);
    }
}

function debounce(fn, delay) {
    let timer;
    return (...args) => {
        clearTimeout(timer);
        timer = setTimeout(() => fn(...args), delay);
    };
}

async function fetchAnthropicModels(key, datalistId) {
    try {
        const res = await fetch('https://api.anthropic.com/v1/models', {
            headers: {
                'x-api-key': key,
                'anthropic-version': '2023-06-01',
                'anthropic-dangerous-direct-browser-access': 'true',
            }
        });
        if (res.ok) {
            const data = await res.json();
            const datalist = document.getElementById(datalistId);
            datalist.innerHTML = '';
            (data.data || []).forEach(m => {
                const opt = document.createElement('option');
                opt.value = m.id;
                datalist.appendChild(opt);
            });
        }
    } catch (e) {
        // Ignore
    }
}

async function testOllamaConnection(hostInputId, statusId) {
    const host = document.getElementById(hostInputId).value.trim() || 'http://localhost:11434';
    const statusEl = document.getElementById(statusId);
    statusEl.textContent = 'Testing...';
    statusEl.className = '';

    try {
        const res = await fetch(`${host}/api/tags`, { signal: AbortSignal.timeout(5000) });
        if (res.ok) {
            const data = await res.json();
            statusEl.textContent = '✓ Connected';
            statusEl.className = 'status-ok';
            return data.models || [];
        } else {
            statusEl.textContent = '✗ Connection failed';
            statusEl.className = 'status-err';
            return [];
        }
    } catch (e) {
        statusEl.textContent = '✗ Cannot reach Ollama';
        statusEl.className = 'status-err';
        return [];
    }
}



// ==========================================
// Main Render
// ==========================================
function renderMain() {
    document.getElementById('mainContent').style.display = '';
    document.getElementById('mainTabs').style.display = '';
    renderProviders();
    renderCognition();
}

// ==========================================
// Tab switching
// ==========================================
document.getElementById('mainTabs').addEventListener('click', (e) => {
    const tab = e.target.closest('.tab');
    if (!tab) return;
    const tabName = tab.dataset.tab;
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    document.getElementById(`tab-${tabName}`).classList.add('active');

    // Load content when tabs are clicked
    if (tabName === 'voice') {
        renderVoice();
    } else if (tabName === 'embodiment') {
        loadEmbodiment();
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
    el.innerHTML = providers.map(p => `
        <div class="provider-card" data-id="${p.id}">
            <div class="provider-info">
                <div class="provider-name">${escapeHtml(p.name)}</div>
                <div class="provider-meta">
                    <span class="provider-platform-badge badge-${p.platform}">${p.platform}</span>
                    ${escapeHtml(p.model)}
                    ${p.host ? ` · ${escapeHtml(p.host)}` : ''}
                </div>
            </div>
            <div class="provider-actions">
                <button class="btn btn-secondary" onclick="openEditModal(${p.id})">Edit</button>
                <button class="btn btn-danger" onclick="confirmDelete(${p.id}, '${escapeHtml(p.name).replace(/'/g, "\\'")}')">Delete</button>
            </div>
        </div>
    `).join('');
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

    if (id) {
        const p = providers.find(x => x.id === id);
        if (p) {
            editPlatform = p.platform;
            document.getElementById('editName').value = p.name;
            document.getElementById('editModel').value = p.model;
            if (p.host) document.getElementById('editHost').value = p.host;
        }
    } else {
        editPlatform = 'ollama';
    }

    // Clear any previous test result
    const testResult = document.getElementById('testResult');
    testResult.className = 'test-result hidden';
    testResult.innerHTML = '';

    selectPlatform(editPlatform, 'edit');
    modal.classList.remove('hidden');
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

// Edit test connection (Ollama host quick-check)
document.getElementById('editTestConnectionBtn').addEventListener('click', async () => {
    const models = await testOllamaConnection('editHost', 'editConnectionStatus');
    if (models.length > 0) {
        const datalist = document.getElementById('editModelSuggestions');
        datalist.innerHTML = '';
        models.forEach(m => {
            const opt = document.createElement('option');
            opt.value = m.name || m.model || m;
            datalist.appendChild(opt);
        });
    }
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

    const body = {
        platform,
        model: document.getElementById('editModel').value.trim(),
    };

    if (id) body.provider_id = id;
    if (config.hasHost) body.host = document.getElementById('editHost').value.trim() || 'http://localhost:11434';
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

    const body = {
        name: document.getElementById('editName').value.trim(),
        platform: platform,
        model: document.getElementById('editModel').value.trim(),
    };

    if (config.hasHost) {
        body.host = document.getElementById('editHost').value.trim() || 'http://localhost:11434';
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
        const data = await res.json();
        if (id) {
            providers = providers.map(p => p.id === id ? data.provider : p);
        } else {
            providers.push(data.provider);
        }
        document.getElementById('providerModal').classList.add('hidden');
        renderProviders();
        renderCognition();
        showToast(id ? 'Provider updated' : 'Provider added', 'success');
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
        renderCognition();
        showToast('Provider deleted', 'success');
    } else {
        const err = await res.json();
        showToast(err.error || 'Cannot delete provider', 'error');
        document.getElementById('deleteModal').classList.add('hidden');
    }
});

// ==========================================
// Cognition Tab
// ==========================================
async function loadAssignments() {
    try {
        const res = await apiFetch('/providers/jobs');
        if (res.ok) {
            const data = await res.json();
            assignments = {};
            (data.assignments || []).forEach(a => {
                assignments[a.job_name] = a.provider_id;
            });
        }
    } catch (e) {
        // ignore
    }
}

function renderCognition() {
    const el = document.getElementById('cognitionList');
    if (providers.length === 0) {
        el.innerHTML = '<div class="empty-state"><h3>No providers configured</h3><p>Add a provider first.</p></div>';
        return;
    }

    el.innerHTML = JOBS.map(job => {
        const currentAssignment = assignments[job.id];
        const options = providers.map(p =>
            `<option value="${p.id}" ${p.id === currentAssignment ? 'selected' : ''}>${escapeHtml(p.name)}</option>`
        ).join('');

        return `
            <div class="job-card">
                <div class="job-info">
                    <div class="job-name">${escapeHtml(job.name)}</div>
                    <div class="job-desc">${escapeHtml(job.desc)}</div>
                </div>
                <span class="job-badge ${job.badgeClass}">${escapeHtml(job.badge)}</span>
                <div class="job-assign">
                    <select class="provider-select" data-job="${job.id}" onchange="assignJob('${job.id}', this)">
                        <option value="">-- Select provider --</option>
                        ${options}
                    </select>
                    <span class="save-indicator" id="save-${job.id}">Saved ✓</span>
                </div>
            </div>
        `;
    }).join('');
}

async function assignJob(jobName, selectEl) {
    const providerId = parseInt(selectEl.value);
    if (!providerId) return;

    try {
        const res = await apiFetch(`/providers/jobs/${jobName}`, {
            method: 'PUT',
            body: JSON.stringify({ provider_id: providerId }),
        });

        if (res.ok) {
            assignments[jobName] = providerId;
            const indicator = document.getElementById(`save-${jobName}`);
            indicator.classList.add('visible');
            setTimeout(() => indicator.classList.remove('visible'), 2000);
        } else {
            showToast('Failed to save assignment', 'error');
        }
    } catch (e) {
        showToast('Network error', 'error');
    }
}

// ==========================================
// Voice Configuration Tab
// ==========================================
async function renderVoice() {
    const el = document.getElementById('voiceConfig');

    try {
        const res = await apiFetch('/system/voice-config');
        if (!res.ok) throw new Error('Failed to load voice config');

        const cfg = await res.json();
        const ttsEndpoint = cfg.tts_endpoint || '';
        const sttEndpoint = cfg.stt_endpoint || '';

        el.innerHTML = `
            <div class="voice-form">
                <div class="form-group">
                    <label>Text-to-Speech Endpoint (OpenAI-compatible)</label>
                    <input type="text" id="ttsEndpoint" placeholder="https://tts.example.com/v1/audio/speech" value="${escapeHtml(ttsEndpoint)}">
                    <p class="form-hint">POST request with <code>{"text": "..."}</code>, returns binary audio (mp3/wav/ogg)</p>
                    <button class="btn btn-secondary" id="testTtsBtn" style="margin-top: 8px;">Test TTS</button>
                    <span id="ttsStatus" style="margin-left: 8px;"></span>
                </div>

                <div class="form-group">
                    <label>Speech-to-Text Endpoint (OpenAI-compatible)</label>
                    <input type="text" id="sttEndpoint" placeholder="https://stt.example.com/v1/audio/transcriptions" value="${escapeHtml(sttEndpoint)}">
                    <p class="form-hint">POST request with multipart form field <code>file</code> (WAV audio), returns <code>{"text": "..."}</code></p>
                    <button class="btn btn-secondary" id="testSttBtn" style="margin-top: 8px;">Test STT</button>
                    <span id="sttStatus" style="margin-left: 8px;"></span>
                </div>

                <div class="form-actions">
                    <button class="btn btn-primary" id="saveVoiceBtn">Save Voice Config</button>
                </div>
            </div>
        `;

        document.getElementById('saveVoiceBtn').addEventListener('click', saveVoiceConfig);
        document.getElementById('testTtsBtn').addEventListener('click', testTtsEndpoint);
        document.getElementById('testSttBtn').addEventListener('click', testSttEndpoint);
    } catch (e) {
        el.innerHTML = `<div class="empty-state"><h3>Error loading voice config</h3><p>${escapeHtml(e.message)}</p></div>`;
    }
}

async function saveVoiceConfig() {
    const ttsEndpoint = document.getElementById('ttsEndpoint').value.trim();
    const sttEndpoint = document.getElementById('sttEndpoint').value.trim();

    try {
        const res = await apiFetch('/system/voice-config', {
            method: 'PUT',
            body: JSON.stringify({ tts_endpoint: ttsEndpoint, stt_endpoint: sttEndpoint }),
        });

        if (res.ok) {
            showToast('Voice config saved', 'success');
        } else {
            showToast('Failed to save voice config', 'error');
        }
    } catch (e) {
        showToast('Network error', 'error');
    }
}

async function testTtsEndpoint() {
    const endpoint = document.getElementById('ttsEndpoint').value.trim();
    const status = document.getElementById('ttsStatus');

    if (!endpoint) {
        status.textContent = '⚠ Enter endpoint first';
        status.style.color = 'orange';
        return;
    }

    status.textContent = '⏳ Testing...';
    status.style.color = 'gray';

    try {
        const res = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: 'Hello, this is a test.' }),
        });

        if (res.ok) {
            status.textContent = '✓ Success';
            status.style.color = 'green';
        } else {
            status.textContent = `✗ Error ${res.status}`;
            status.style.color = 'red';
        }
    } catch (e) {
        status.textContent = '✗ Connection failed';
        status.style.color = 'red';
    }

    setTimeout(() => {
        status.textContent = '';
    }, 3000);
}

async function testSttEndpoint() {
    const endpoint = document.getElementById('sttEndpoint').value.trim();
    const status = document.getElementById('sttStatus');

    if (!endpoint) {
        status.textContent = '⚠ Enter endpoint first';
        status.style.color = 'orange';
        return;
    }

    status.textContent = '⏳ Testing...';
    status.style.color = 'gray';

    try {
        // Create a simple silence WAV file for testing
        const wav = createSilenceWav();
        const formData = new FormData();
        formData.append('file', wav, 'test.wav');

        const res = await fetch(endpoint, {
            method: 'POST',
            body: formData,
        });

        if (res.ok) {
            status.textContent = '✓ Success';
            status.style.color = 'green';
        } else {
            status.textContent = `✗ Error ${res.status}`;
            status.style.color = 'red';
        }
    } catch (e) {
        status.textContent = '✗ Connection failed';
        status.style.color = 'red';
    }

    setTimeout(() => {
        status.textContent = '';
    }, 3000);
}

function createSilenceWav() {
    const sampleRate = 16000;
    const duration = 0.5; // 500ms of silence
    const numSamples = sampleRate * duration;
    const dataLength = numSamples * 2; // 16-bit = 2 bytes/sample
    const buffer = new ArrayBuffer(44 + dataLength);
    const view = new DataView(buffer);

    const writeStr = (offset, str) => {
        for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
    };

    writeStr(0, 'RIFF');
    view.setUint32(4, 36 + dataLength, true);
    writeStr(8, 'WAVE');
    writeStr(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeStr(36, 'data');
    view.setUint32(40, dataLength, true);

    return new Blob([buffer], { type: 'audio/wav' });
}

// ==========================================
// Embodiment Tab — Tool Management
// ==========================================
async function loadEmbodiment() {
    try {
        const res = await apiFetch('/tools');
        if (!res.ok) throw new Error('Failed to load tools');
        const data = await res.json();
        embodyTools = data.tools || [];
        renderTools('');

        // Start polling if any tool is building
        if (embodyTools.some(t => t.status === 'building')) {
            startBuildPoll();
        }
    } catch (e) {
        console.error('loadEmbodiment error:', e);
        document.getElementById('toolsGrid').innerHTML = `<div class="empty-state"><h3>Error loading tools</h3><p>${escapeHtml(e.message)}</p></div>`;
    }
}

function renderTools(filter = '') {
    const grid = document.getElementById('toolsGrid');
    if (!grid) return;

    let filtered = embodyTools;
    if (filter.trim()) {
        const q = filter.toLowerCase();
        filtered = embodyTools.filter(t =>
            t.name.toLowerCase().includes(q) ||
            t.description.toLowerCase().includes(q) ||
            (t.category && t.category.toLowerCase().includes(q))
        );
    }

    if (embodyTools.length === 0) {
        grid.innerHTML = `
            <div class="tools-empty" style="grid-column: 1/-1; text-align: center; padding: 64px 24px;">
                <h3>No tools installed yet</h3>
                <p>Chalie can gain new abilities by installing tools.</p>
                <button class="btn btn-primary" onclick="openInstallModal()" style="margin-top: 16px;">+ Add Tool</button>
            </div>
        `;
        return;
    }

    if (filtered.length === 0) {
        grid.innerHTML = `<div class="tools-empty" style="grid-column: 1/-1; text-align: center; padding: 32px 0;"><p>No results for "${escapeHtml(filter)}"</p></div>`;
        return;
    }

    grid.innerHTML = filtered.map(t => renderToolCard(t)).join('');
}

function renderToolCard(tool) {
    const name = tool.name;
    const icon = tool.icon || '⚙';
    const status = tool.status;
    const hasError = status === 'error';
    const isBuilding = status === 'building';
    const isDisabled = status === 'disabled';
    const hasConfig = (tool.config_schema || []).length > 0;

    // Status badge styling
    const statusBadgeMap = {
        'connected': { label: 'Active', class: '--connected' },
        'available': { label: 'Ready', class: '--available' },
        'system': { label: 'System', class: '--system' },
        'disabled': { label: 'Disabled', class: '--disabled' },
        'building': { label: '⏳ Building…', class: '--building' },
        'error': { label: 'Error', class: '--error' },
    };
    const statusInfo = statusBadgeMap[status] || { label: status, class: '' };

    // Actions HTML
    let actionsHtml = '';
    if (!isBuilding) {
        if (hasConfig && !isDisabled) {
            actionsHtml += `<button class="tool-card__btn" onclick="openToolSettings('${name}')" title="Settings">⚙ Settings</button>`;
        }
        if (isDisabled) {
            actionsHtml += `<button class="tool-card__btn --primary" onclick="enableTool('${name}')">Enable</button>`;
        } else {
            actionsHtml += `<button class="tool-card__btn --danger" onclick="disableTool('${name}')">Disable</button>`;
        }
    }

    // Error details (collapsible)
    let errorHtml = '';
    if (hasError && tool.last_error) {
        errorHtml = `
            <div class="tool-card__error-row">
                <button class="tool-card__error-toggle" onclick="toggleErrorDetails(this)">▶ Error details</button>
                <div class="tool-card__error-details hidden">
                    <div class="tool-card__error-msg">${escapeHtml(tool.last_error)}</div>
                </div>
            </div>
        `;
    }

    return `
        <div class="tool-card ${isBuilding ? '--building' : ''} ${isDisabled ? '--disabled' : ''} ${hasError ? '--error' : ''}">
            <div class="tool-card__header">
                <div class="tool-card__icon">${renderIconHtml(icon)}</div>
                <div>
                    <div class="tool-card__name">${escapeHtml(tool.display_name || tool.name)}</div>
                    ${tool.category ? `<div class="tool-card__category">${escapeHtml(tool.category)}</div>` : ''}
                </div>
            </div>
            <p class="tool-card__desc">${escapeHtml(tool.description)}</p>
            ${errorHtml}
            <div class="tool-card__footer">
                <span class="tool-card__status ${statusInfo.class}">${statusInfo.label}</span>
                <div class="tool-card__actions">
                    ${actionsHtml}
                </div>
            </div>
        </div>
    `;
}

function toggleErrorDetails(btn) {
    const details = btn.nextElementSibling;
    if (details.classList.contains('hidden')) {
        details.classList.remove('hidden');
        btn.textContent = '▼ Error details';
    } else {
        details.classList.add('hidden');
        btn.textContent = '▶ Error details';
    }
}

function startBuildPoll() {
    if (pollTimer) return;
    pollTimer = setInterval(async () => {
        if (document.hidden) return; // pause when tab not visible
        await loadEmbodiment();
        const stillBuilding = embodyTools.some(t => t.status === 'building');
        if (!stillBuilding) {
            clearInterval(pollTimer);
            pollTimer = null;
        }
    }, 3000);
}

function openInstallModal() {
    document.getElementById('installModal').classList.remove('hidden');
    document.getElementById('installGitUrl').value = '';
    document.getElementById('installZipFile').value = '';
    document.getElementById('installProgress').classList.add('hidden');
    document.getElementById('installError').classList.add('hidden');
    installMethod = 'git';
    selectInstallMethod('git');
}

function closeInstallModal() {
    document.getElementById('installModal').classList.add('hidden');
}

function selectInstallMethod(method) {
    installMethod = method;
    document.querySelectorAll('#installMethodTabs .platform-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.installMethod === method);
    });
    document.getElementById('installGitPanel').classList.toggle('hidden', method !== 'git');
    document.getElementById('installZipPanel').classList.toggle('hidden', method !== 'zip');
}

async function handleInstall() {
    const errorEl = document.getElementById('installError');
    const progressEl = document.getElementById('installProgress');
    const btn = document.getElementById('installBtn');

    errorEl.classList.add('hidden');
    progressEl.classList.remove('hidden');

    let formData;

    if (installMethod === 'git') {
        const url = document.getElementById('installGitUrl').value.trim();
        if (!url) {
            showError(errorEl, 'Enter a repository URL');
            return;
        }
        progressEl.innerHTML = 'Cloning repository…';
        formData = JSON.stringify({ git_url: url });
    } else {
        const fileInput = document.getElementById('installZipFile');
        if (!fileInput.files.length) {
            showError(errorEl, 'Select a ZIP file');
            return;
        }
        progressEl.innerHTML = 'Uploading…';
        formData = new FormData();
        formData.append('zip_file', fileInput.files[0]);
    }

    btn.disabled = true;

    try {
        const res = await apiFetch('/tools/install', {
            method: 'POST',
            body: formData,
        }, installMethod === 'zip');

        const data = await res.json();

        if (data.ok) {
            progressEl.innerHTML = 'Building container…';
            setTimeout(() => {
                closeInstallModal();
                loadEmbodiment();
                startBuildPoll();
                showToast(`Tool "${data.tool_name}" installing…`, 'success');
            }, 1500);
        } else {
            showError(errorEl, data.error || 'Installation failed');
            btn.disabled = false;
        }
    } catch (e) {
        showError(errorEl, `Network error: ${e.message}`);
        btn.disabled = false;
    }
}

function showError(el, msg) {
    el.textContent = msg;
    el.classList.remove('hidden');
}

async function disableTool(name) {
    try {
        const res = await apiFetch(`/tools/${name}/disable`, { method: 'POST' });
        if (res.ok) {
            showToast(`Tool disabled`, 'success');
            await loadEmbodiment();
        } else {
            const err = await res.json();
            showToast(err.error || 'Failed to disable tool', 'error');
        }
    } catch (e) {
        showToast('Error: ' + e.message, 'error');
    }
}

async function enableTool(name) {
    try {
        const res = await apiFetch(`/tools/${name}/enable`, { method: 'POST' });
        const data = await res.json();
        if (data.ok) {
            showToast(`Enabling tool… building container`, 'success');
            await loadEmbodiment();
            startBuildPoll();
        } else {
            showToast(data.error || 'Failed to enable tool', 'error');
        }
    } catch (e) {
        showToast('Error: ' + e.message, 'error');
    }
}

async function openToolSettings(name) {
    try {
        const res = await apiFetch(`/tools/${name}/config`);
        if (!res.ok) throw new Error('Failed to load config');
        const data = await res.json();

        settingsTool = name;
        document.getElementById('toolSettingsTitle').textContent = `${escapeHtml(name)} Settings`;

        const schema = data.config_schema || {};
        const config = data.config || {};

        let formHtml = '';
        for (const [key, fieldDef] of Object.entries(schema)) {
            const value = config[key] || '';
            const isSecret = fieldDef.secret;
            const isMultiline = fieldDef.multiline;
            const hint = fieldDef.hint || '';

            if (isMultiline) {
                formHtml += `
                    <div class="form-group">
                        <label>${escapeHtml(fieldDef.label || key)}</label>
                        <textarea id="config_${key}"
                                  rows="5"
                                  placeholder="${escapeHtml(fieldDef.placeholder || '')}">${escapeHtml(value)}</textarea>
                        ${hint ? `<p class="form-hint">${escapeHtml(hint)}</p>` : ''}
                    </div>
                `;
            } else {
                formHtml += `
                    <div class="form-group">
                        <label>${escapeHtml(fieldDef.label || key)}</label>
                        <input type="${isSecret ? 'password' : 'text'}"
                               id="config_${key}"
                               value="${escapeHtml(value)}"
                               placeholder="${escapeHtml(fieldDef.placeholder || '')}"
                               data-secret="${isSecret}">
                        ${hint ? `<p class="form-hint">${escapeHtml(hint)}</p>` : ''}
                    </div>
                `;
            }
        }

        if (!formHtml) {
            formHtml = '<p style="color: var(--text-muted); font-size: 13px;">No configuration needed for this tool.</p>';
        } else {
            formHtml += '<p style="color: var(--text-muted); font-size: 12px; margin-top: 12px;"><em>Configuration is stored securely and encrypted.</em></p>';
        }

        document.getElementById('toolSettingsForm').innerHTML = formHtml;
        document.getElementById('toolSettingsModal').classList.remove('hidden');
    } catch (e) {
        showToast('Error loading config: ' + e.message, 'error');
    }
}

async function saveToolSettings() {
    if (!settingsTool) return;

    const config = {};
    const inputs = document.querySelectorAll('#toolSettingsForm input[id^="config_"]');
    inputs.forEach(inp => {
        const key = inp.id.replace('config_', '');
        config[key] = inp.value;
    });
    const textareas = document.querySelectorAll('#toolSettingsForm textarea[id^="config_"]');
    textareas.forEach(ta => {
        const key = ta.id.replace('config_', '');
        config[key] = ta.value;
    });

    try {
        const res = await apiFetch(`/tools/${settingsTool}/config`, {
            method: 'PUT',
            body: JSON.stringify(config),
        });

        if (res.ok) {
            document.getElementById('toolSettingsModal').classList.add('hidden');
            showToast('Settings saved', 'success');
            await loadEmbodiment();
        } else {
            const err = await res.json();
            showToast(err.error || 'Failed to save settings', 'error');
        }
    } catch (e) {
        showToast('Error: ' + e.message, 'error');
    }
}

// ==========================================
// Embodiment Event Listeners
// ==========================================
document.getElementById('addToolBtn').addEventListener('click', openInstallModal);
document.getElementById('closeInstallModal').addEventListener('click', closeInstallModal);
document.getElementById('cancelInstallBtn').addEventListener('click', closeInstallModal);
document.getElementById('installBtn').addEventListener('click', handleInstall);
document.getElementById('toolSearch').addEventListener('input', (e) => {
    renderTools(e.target.value);
});

document.getElementById('installMethodTabs').addEventListener('click', (e) => {
    const tab = e.target.closest('.platform-tab');
    if (tab) selectInstallMethod(tab.dataset.installMethod);
});

document.getElementById('closeToolSettingsModal').addEventListener('click', () => {
    document.getElementById('toolSettingsModal').classList.add('hidden');
    settingsTool = null;
});

document.getElementById('cancelToolSettingsBtn').addEventListener('click', () => {
    document.getElementById('toolSettingsModal').classList.add('hidden');
    settingsTool = null;
});

document.getElementById('saveToolSettingsBtn').addEventListener('click', saveToolSettings);

document.getElementById('browseMarketplaceBtn').addEventListener('click', (e) => {
    e.preventDefault();
    showToast('Chalie Marketplace coming soon!', 'info');
});

// ==========================================
// Helpers
// ==========================================
function escapeHtml(str) {
    if (!str) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

function renderIconHtml(icon) {
    if (!icon) return '&#x2699;';
    if (icon.startsWith('http://') || icon.startsWith('https://') || icon.startsWith('/')) {
        return `<img src="${escapeHtml(icon)}" style="width:100%;height:100%;object-fit:contain" alt="">`;
    }
    if (icon.startsWith('fa-')) {
        return `<i class="fa-solid ${escapeHtml(icon)}"></i>`;
    }
    return escapeHtml(icon);
}

// ==========================================
// Start
// ==========================================
init();
