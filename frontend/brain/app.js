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

// Cognition observability state
let obsData = {};               // cached API responses keyed by subtab name
let obsLoaded = {};             // whether a subtab has been fetched
let activeSubtab = 'jobs';      // currently active cognition sub-tab

// ==========================================
// LLM Jobs
// ==========================================
const JOBS = [
    // ── Tier 1: ≥ 30B ──────────────────────────────────────
    { id: 'autobiography',            name: 'Autobiography Synthesis',  desc: 'Synthesises personal narrative prose from all memory layers (6h cycle).',               badge: '≥ 30B', badgeClass: 'badge-30b', tokens: '~7.7K',    frequency: 'Every 6 hours',            strengths: ['Reasoning', 'Creative Writing', 'Synthesis'] },
    { id: 'frontal-cortex',           name: 'Frontal Cortex',           desc: 'Core reasoning engine; orchestrates all response modes.',                                badge: '≥ 30B', badgeClass: 'badge-30b', tokens: '~5K',      frequency: 'Once per message',          strengths: ['Reasoning', 'Structured Output', 'Context Following'] },
    { id: 'frontal-cortex-act',       name: 'Act Mode',                 desc: 'Plans and executes multi-step tool actions. Up to 7 iterations per invocation.',          badge: '≥ 30B', badgeClass: 'badge-30b', tokens: '~5.4K ×N', frequency: 'Per message (ACT mode)',    strengths: ['Strong Reasoning', 'Structured Output', 'Planning'] },
    { id: 'frontal-cortex-respond',   name: 'Respond Mode',             desc: 'Primary conversational voice of Chalie in normal conversation.',                          badge: '≥ 30B', badgeClass: 'badge-30b', tokens: '~5.4K',    frequency: 'Once per message',          strengths: ['Reasoning', 'Structured Output', 'Natural Language'] },

    // ── Tier 2: ≥ 14B ──────────────────────────────────────
    { id: 'cognitive-drift',          name: 'Cognitive Drift (DMN)',     desc: 'Generates spontaneous thoughts during idle windows (Default Mode Network).',              badge: '≥ 14B', badgeClass: 'badge-14b', tokens: '~1.2K',    frequency: 'Idle (every 5–10 min)',    strengths: ['Reasoning', 'Creativity'] },
    { id: 'episodic-memory',          name: 'Episodic Memory',           desc: 'Synthesises sessions into episodic narratives for long-term recall.',                     badge: '≥ 14B', badgeClass: 'badge-14b', tokens: '~6.8K',    frequency: 'Batch consolidation',      strengths: ['Reasoning', 'Structured Output', 'Narrative Synthesis'] },
    { id: 'frontal-cortex-clarify',   name: 'Clarify Mode',             desc: 'Asks clarifying questions when user intent is ambiguous.',                                badge: '≥ 14B', badgeClass: 'badge-14b', tokens: '~2.8K',    frequency: 'Per message (CLARIFY)',     strengths: ['Reasoning', 'Structured Output'] },
    { id: 'frontal-cortex-proactive', name: 'Proactive Mode',           desc: 'Translates spontaneous thoughts into outreach messages.',                                 badge: '≥ 14B', badgeClass: 'badge-14b', tokens: '~3K',      frequency: 'Idle triggered',           strengths: ['Reasoning', 'Natural Language', 'Structured Output'] },
    { id: 'mode-reflection',          name: 'Mode Reflection',          desc: 'Peer-reviews routing decisions during idle time (nightly batch).',                         badge: '≥ 14B', badgeClass: 'badge-14b', tokens: '~1.5K',    frequency: 'Nightly batch',            strengths: ['Reasoning', 'Structured Output', 'Analysis'] },
    { id: 'semantic-memory',          name: 'Semantic Memory',           desc: 'Extracts concepts and relationships to build the knowledge graph.',                       badge: '≥ 14B', badgeClass: 'badge-14b', tokens: '~5.6K',    frequency: 'Per exchange (async)',      strengths: ['Reasoning', 'Structured Output', 'Knowledge Extraction'] },

    // ── Tier 3: 8B sufficient ───────────────────────────────
    { id: 'cognitive-triage',           name: 'Cognitive Triage',          desc: 'Routes user input to optimal cognitive branch (RESPOND/CLARIFY/ACT). Lightweight preferred.', badge: '8B sufficient', badgeClass: 'badge-8b', tokens: '~2.6K', frequency: 'Once per message',      strengths: ['Structured Output', 'Classification'] },
    { id: 'experience-assimilation',    name: 'Experience Assimilation',   desc: 'Evaluates tool outputs for novel knowledge worth storing.',                               badge: '8B sufficient', badgeClass: 'badge-8b', tokens: '~2.4K', frequency: 'Post-tool execution',   strengths: ['Structured Output', 'Classification'] },
    { id: 'fact-store',                 name: 'Fact Store',                desc: 'Extracts and stores atomic facts from exchanges. Runs async.',                            badge: '8B sufficient', badgeClass: 'badge-8b', tokens: '~1.5K', frequency: 'Per exchange (async)',   strengths: ['Structured Output', 'Extraction'] },
    { id: 'frontal-cortex-acknowledge', name: 'Acknowledge Mode',          desc: 'Brief acknowledgments for greetings and simple inputs.',                                  badge: '8B sufficient', badgeClass: 'badge-8b', tokens: '~2.1K', frequency: 'Per message (ACK mode)', strengths: ['Structured Output'] },
    { id: 'memory-chunker',             name: 'Memory Chunker',            desc: 'Extracts gists, facts, and traits from exchanges. Runs async.',                           badge: '8B sufficient', badgeClass: 'badge-8b', tokens: '~4.1K', frequency: 'Per exchange (async)',   strengths: ['Structured Output', 'Extraction'] },
    { id: 'moment-enrichment',          name: 'Moment Enrichment',         desc: 'Generates titles and summaries for pinned moments. Runs in a 5-minute background poll.',   badge: '8B sufficient', badgeClass: 'badge-8b', tokens: '~300',  frequency: 'Per pinned moment',     strengths: ['Summarisation', 'Extraction'] },

    // ── Tier 4: 4B sufficient ───────────────────────────────
    { id: 'mode-tiebreaker', name: 'Mode Tiebreaker', desc: 'Resolves ambiguous routing with binary A-vs-B decision. Must be fast.', badge: '4B sufficient', badgeClass: 'badge-4b', tokens: '~600',  frequency: '<5% of messages',    strengths: ['Fast Inference', 'Classification'] },
    { id: 'topic-namer',     name: 'Topic Namer',     desc: 'Generates short display names for conversation topics.',               badge: '4B sufficient', badgeClass: 'badge-4b', tokens: '~550',  frequency: '5–10% of messages',  strengths: ['Fast Inference'] },
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

    // Handle OAuth callback URL parameters
    handleOAuthCallback();
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
    if (tabName === 'cognition') {
        loadCognitionSubtab(activeSubtab);
    } else if (tabName === 'embodiment') {
        loadEmbodiment();
    } else if (tabName === 'scheduler') {
        loadScheduler();
    } else if (tabName === 'lists') {
        loadLists();
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

        const strengthTags = (job.strengths || []).map(s =>
            `<span class="job-strength">${escapeHtml(s)}</span>`
        ).join('');

        return `
            <div class="job-card">
                <div class="job-card__top">
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
                <div class="job-card__meta">
                    <span class="job-meta-item" title="Average tokens per invocation">
                        <i class="fa-solid fa-bolt job-meta-icon"></i> ${escapeHtml(job.tokens)} tokens
                    </span>
                    <span class="job-meta-item" title="Usage frequency">
                        <i class="fa-regular fa-clock job-meta-icon"></i> ${escapeHtml(job.frequency)}
                    </span>
                    <span class="job-meta-sep"></span>
                    <div class="job-strengths">${strengthTags}</div>
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

    // OAuth status
    const hasOAuth = tool.auth_type === 'oauth2';
    const oauthConnected = tool.oauth_connected;

    // Actions HTML
    let actionsHtml = '';
    if (!isBuilding) {
        if (hasConfig || hasOAuth) {
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

    // OAuth badge HTML
    let oauthBadgeHtml = '';
    if (hasOAuth) {
        if (oauthConnected) {
            oauthBadgeHtml = `<span class="oauth-badge --connected">${escapeHtml(tool.auth_provider_hint || 'OAuth')} Connected</span>`;
        } else {
            oauthBadgeHtml = `<span class="oauth-badge --disconnected">Not Connected</span>`;
        }
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
            ${oauthBadgeHtml}
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
                                  placeholder="${escapeHtml(fieldDef.placeholder || '')}"
                                  data-secret="${isSecret}">${escapeHtml(value)}</textarea>
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

        // OAuth section — generic, reads auth_type from tool listing data
        const toolInfo = embodyTools.find(t => t.name === name);
        if (toolInfo && toolInfo.auth_type === 'oauth2') {
            const providerHint = toolInfo.auth_provider_hint || 'OAuth';
            const hasCredentials = !!(config.client_id && config.client_id !== '***' && config.client_id !== '') ||
                                   !!(config.client_secret && config.client_secret === '***');

            formHtml += `<div class="oauth-section" style="margin-top:20px;padding-top:16px;border-top:1px solid var(--border)">`;

            // Setup guide when credentials are empty
            if (!hasCredentials) {
                const callbackUrl = `${window.location.origin}/tools/${name}/oauth/callback`;
                formHtml += `
                    <div class="oauth-setup-guide">
                        <div style="font-size:13px;font-weight:600;color:var(--text);margin-bottom:12px">
                            Setup ${escapeHtml(providerHint)} Credentials
                        </div>
                        <ol style="font-size:12px;color:var(--text-muted);line-height:1.8;padding-left:18px;margin:0 0 12px 0">
                            <li>Go to <a href="https://console.cloud.google.com" target="_blank" rel="noopener" style="color:var(--accent-hover)">console.cloud.google.com</a></li>
                            <li>Create a new project (or select an existing one)</li>
                            <li>Enable APIs:
                                <a href="https://console.cloud.google.com/apis/library/gmail.googleapis.com" target="_blank" rel="noopener" style="color:var(--accent-hover)">Gmail</a>,
                                <a href="https://console.cloud.google.com/apis/library/calendar-json.googleapis.com" target="_blank" rel="noopener" style="color:var(--accent-hover)">Calendar</a>,
                                <a href="https://console.cloud.google.com/apis/library/tasks.googleapis.com" target="_blank" rel="noopener" style="color:var(--accent-hover)">Tasks</a>
                            </li>
                            <li>Go to <a href="https://console.cloud.google.com/apis/credentials" target="_blank" rel="noopener" style="color:var(--accent-hover)">Credentials</a> &rarr; Create OAuth Client ID &rarr; Web application</li>
                            <li>Add redirect URI: <code style="font-size:11px;color:var(--accent-hover);background:rgba(138,92,255,0.08);padding:2px 6px;border-radius:3px">${escapeHtml(callbackUrl)}</code></li>
                            <li>Go to <a href="https://console.cloud.google.com/apis/credentials/consent" target="_blank" rel="noopener" style="color:var(--accent-hover)">OAuth consent screen</a> &rarr; <strong>Test users</strong> &rarr; Add your Google email address <span style="opacity:0.5">(required while app is in "Testing" publishing status)</span></li>
                            <li>Paste Client ID and Client Secret in the fields above, then click Save</li>
                        </ol>
                    </div>`;
            }

            // OAuth connect/disconnect status
            formHtml += `<div id="oauthStatusArea" style="margin-top:12px">`;
            if (toolInfo.oauth_connected) {
                formHtml += `
                    <div class="oauth-status --connected">
                        <i class="fas fa-check-circle" style="color:var(--success);margin-right:6px"></i>
                        <span>${escapeHtml(providerHint)} account connected</span>
                        <button class="tool-card__btn --danger" style="margin-left:auto" onclick="disconnectOAuth('${name}')">Disconnect</button>
                    </div>`;
            } else if (hasCredentials) {
                formHtml += `
                    <div class="oauth-status --ready">
                        <span style="color:var(--text-muted);font-size:13px">Account not connected</span>
                        <button class="btn btn-primary" style="margin-left:auto;font-size:12px;padding:6px 14px" onclick="startOAuth('${name}')">
                            Connect ${escapeHtml(providerHint)} Account
                        </button>
                    </div>`;
            } else {
                formHtml += `
                    <div class="oauth-status --pending">
                        <span style="color:var(--text-muted);font-size:12px;font-style:italic">
                            Save your credentials above, then connect your account
                        </span>
                    </div>`;
            }
            formHtml += `</div></div>`;
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
        // Skip secret fields still showing the server-side mask — don't overwrite stored value
        if (inp.dataset.secret === 'true' && inp.value === '***') return;
        config[key] = inp.value;
    });
    const textareas = document.querySelectorAll('#toolSettingsForm textarea[id^="config_"]');
    textareas.forEach(ta => {
        const key = ta.id.replace('config_', '');
        if (ta.dataset.secret === 'true' && ta.value === '***') return;
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
// OAuth Functions (generic, tool-agnostic)
// ==========================================
async function startOAuth(toolName) {
    try {
        const res = await apiFetch(`/tools/${toolName}/oauth/start`);
        if (!res.ok) {
            const err = await res.json();
            showToast(err.error || 'Failed to start OAuth', 'error');
            return;
        }
        const data = await res.json();
        if (data.auth_url) {
            // Open OAuth URL in popup window
            const popup = window.open(data.auth_url, 'oauth_popup', 'width=600,height=700,scrollbars=yes');
            if (!popup) {
                // Popup blocked — redirect in same window
                window.location.href = data.auth_url;
            }
        }
    } catch (e) {
        showToast('Error starting OAuth: ' + e.message, 'error');
    }
}

async function disconnectOAuth(toolName) {
    try {
        const res = await apiFetch(`/tools/${toolName}/oauth/disconnect`, { method: 'POST' });
        if (res.ok) {
            showToast('Account disconnected', 'success');
            // Close settings modal and reload tools
            document.getElementById('toolSettingsModal').classList.add('hidden');
            settingsTool = null;
            await loadEmbodiment();
        } else {
            const err = await res.json();
            showToast(err.error || 'Failed to disconnect', 'error');
        }
    } catch (e) {
        showToast('Error: ' + e.message, 'error');
    }
}

// Handle OAuth callback parameters on page load
function handleOAuthCallback() {
    const params = new URLSearchParams(window.location.search);
    const oauthSuccess = params.get('oauth_success');
    const oauthError = params.get('oauth_error');
    const toolName = params.get('tool');

    if (oauthSuccess === 'true' && toolName) {
        showToast(`${toolName.replace(/_/g, ' ')} account connected successfully`, 'success');
        // Clean URL
        window.history.replaceState({}, document.title, window.location.pathname);
        // Reload tools to reflect new status
        loadEmbodiment();
        // If this was opened in a popup, close it and refresh parent
        if (window.opener) {
            window.opener.loadEmbodiment();
            window.close();
        }
    } else if (oauthError) {
        showToast(`OAuth error: ${oauthError}`, 'error');
        window.history.replaceState({}, document.title, window.location.pathname);
        if (window.opener) {
            window.opener.showToast(`OAuth error: ${oauthError}`, 'error');
            window.close();
        }
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
        footer.style.display = 'none';
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
    footer.style.display = 'flex';
    loadMoreBtn.style.display = scheduleItems.length < scheduleTotal ? '' : 'none';
    const hasHistory = scheduleItems.some(i => i.status !== 'pending');
    clearBtn.style.display = (scheduleFilter === 'all' || scheduleFilter !== 'pending') && hasHistory ? '' : 'none';
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
        <button class="tool-card__btn" onclick="openEditSchedule('${escapeHtml(item.id)}')">Edit</button>
        <button class="tool-card__btn --danger" onclick="confirmCancelSchedule('${escapeHtml(item.id)}')">Cancel</button>
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
    if (isNaN(date.getTime())) {
        hint.style.display = 'none';
        hint.textContent = '';
        return;
    }

    const DAYS = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
    const dayName = DAYS[date.getDay()];
    const timeStr = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const dom = date.getDate();
    const domSuffix = dom === 1 ? 'st' : dom === 2 ? 'nd' : dom === 3 ? 'rd' : 'th';

    let pattern = '';
    if (recurrence === 'interval') {
        const mins = parseInt(document.getElementById('scheduleIntervalMinutes').value, 10);
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
        const mins = parseInt(document.getElementById('scheduleIntervalMinutes').value, 10);
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
        showToast('Network error', 'error');
    }
}

function formatRecurrence(recurrence) {
    if (!recurrence) return '';
    if (recurrence.startsWith('interval:')) {
        const mins = parseInt(recurrence.split(':')[1], 10);
        if (isNaN(mins)) return recurrence;
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
        <button class="tool-card__btn" onclick="openEditSchedule('${escapeHtml(item.id)}')">Edit</button>
        <button class="tool-card__btn --danger" onclick="confirmCancelSchedule('${escapeHtml(item.id)}')">Cancel</button>
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
    delete obsLoaded[activeSubtab];
    delete obsData[activeSubtab];
    loadCognitionSubtab(activeSubtab);
});

function loadCognitionSubtab(subtab) {
    if (subtab === 'jobs') return; // Jobs panel uses existing renderCognition()
    if (obsLoaded[subtab]) return; // Already cached

    const loaders = {
        routing: loadRoutingObs,
        memory: loadMemoryObs,
        tools: loadToolsObs,
        identity: loadIdentityObs,
        tasks: loadTasksObs,
        understanding: loadUnderstandingObs,
    };
    if (loaders[subtab]) loaders[subtab]();
}

// ── Shared helpers ──

function obsStatCard(label, value, sub) {
    return `<div class="obs-stat-card">
        <span class="obs-stat-card__label">${escapeHtml(label)}</span>
        <span class="obs-stat-card__value">${escapeHtml(String(value))}</span>
        ${sub ? `<span class="obs-stat-card__sub">${escapeHtml(sub)}</span>` : ''}
    </div>`;
}

function obsSkeletonBlock(height) {
    return `<div class="obs-skeleton" style="height:${height}px;margin-bottom:14px"></div>`;
}

function obsSetTimestamp(isoStr) {
    const el = document.getElementById('obsTimestamp');
    if (!el || !isoStr) return;
    try {
        const d = new Date(isoStr);
        el.textContent = 'Updated ' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } catch { el.textContent = ''; }
}

function obsPct(n) {
    return Math.round((n || 0) * 100);
}

// ── Thinking (Routing) ──

async function loadRoutingObs() {
    const el = document.getElementById('routingContent');
    el.innerHTML = obsSkeletonBlock(40) + obsSkeletonBlock(80) + obsSkeletonBlock(120);

    try {
        const res = await apiFetch('/system/observability/routing');
        if (!res.ok) throw new Error('Failed to load');
        const data = await res.json();
        obsData.routing = data;
        obsLoaded.routing = true;
        obsSetTimestamp(data.generated_at);

        const dist = data.distribution || {};
        const respondPct = obsPct(dist['RESPOND']);
        const actPct = obsPct(dist['ACT']);

        let html = `<p class="obs-summary">Over the past week, Chalie chose to respond directly ${respondPct}% of the time and took action ${actPct}% of the time.</p>`;

        // Stat cards
        html += '<div class="obs-stats">';
        html += obsStatCard('Decisions (24h)', data.total_decisions_24h || 0);
        html += obsStatCard('Avg Confidence', ((data.avg_confidence_24h || 0) * 100).toFixed(1) + '%');
        html += obsStatCard('Tiebreaker Rate', ((data.tiebreaker_rate_24h || 0) * 100).toFixed(1) + '%', 'Needed LLM to decide');
        html += '</div>';

        // Mode distribution bar chart
        html += '<div class="obs-section-title">Mode Distribution (7 days)</div>';
        html += '<div class="obs-bar-chart">';
        const modeColors = { RESPOND: '--accent', ACT: '--cyan', CLARIFY: '--warning', ACKNOWLEDGE: '--success', IGNORE: '--error' };
        const sortedModes = Object.entries(dist).sort((a, b) => b[1] - a[1]);
        for (const [mode, pct] of sortedModes) {
            const colorClass = modeColors[mode] || '--accent';
            html += `<div class="obs-bar-row">
                <span class="obs-bar-row__label">${escapeHtml(mode)}</span>
                <div class="obs-bar-row__track"><div class="obs-bar-row__fill ${colorClass}" style="width:${obsPct(pct)}%"></div></div>
                <span class="obs-bar-row__value">${obsPct(pct)}%</span>
            </div>`;
        }
        html += '</div>';

        // Recent decisions
        const recent = data.recent || [];
        if (recent.length > 0) {
            html += `<div class="obs-section-title">Recent Decisions (${recent.length})</div>`;
            html += '<div class="obs-recent-list">';
            for (const d of recent.slice(0, 15)) {
                html += `<div class="obs-recent-item">
                    <span class="obs-recent-item__mode">${escapeHtml(d.mode)}</span>
                    <span class="obs-recent-item__topic">${escapeHtml(d.topic || 'general')}</span>
                    <span class="obs-recent-item__confidence">${(d.confidence * 100).toFixed(0)}%</span>
                </div>`;
            }
            html += '</div>';
        }

        el.innerHTML = html;
    } catch (e) {
        el.innerHTML = '<div class="obs-empty">Could not load thinking data.</div>';
    }
}

// ── Memory ──

async function loadMemoryObs() {
    const el = document.getElementById('memoryContent');
    el.innerHTML = obsSkeletonBlock(40) + obsSkeletonBlock(100);

    try {
        const res = await apiFetch('/system/observability/memory');
        if (!res.ok) throw new Error('Failed to load');
        const data = await res.json();
        obsData.memory = data;
        obsLoaded.memory = true;
        obsSetTimestamp(data.generated_at);

        let html = `<p class="obs-summary">Chalie remembers ${data.episodes || 0} episodes and ${data.concepts || 0} concepts, with ${data.facts || 0} facts in short-term memory.</p>`;

        // Long-term stat cards
        html += '<div class="obs-section-title">Long-Term Memory</div>';
        html += '<div class="obs-stats">';
        html += obsStatCard('Episodes', data.episodes || 0);
        html += obsStatCard('Concepts', data.concepts || 0);
        html += obsStatCard('Traits', data.traits || 0);
        html += '</div>';

        // Health indicators
        html += '<div class="obs-section-title">Health</div>';
        html += '<div class="obs-stats">';
        html += obsStatCard('Avg Episode Activation', (data.avg_episode_activation || 0).toFixed(3), 'Higher = more accessible');
        html += obsStatCard('Avg Trait Strength', (data.avg_trait_strength || 0).toFixed(3), 'Higher = more confident');
        html += '</div>';

        // Short-term memory
        html += '<div class="obs-section-title">Short-Term Memory</div>';
        html += '<div class="obs-stats">';
        html += obsStatCard('Working Memory', data.working_memory || 0, 'Active conversation turns');
        html += obsStatCard('Gists', data.gists || 0, 'Compressed summaries');
        html += obsStatCard('Facts', data.facts || 0, 'Atomic assertions');
        html += '</div>';

        // Queue depths (only if non-zero)
        const queues = data.queues || {};
        const nonZeroQueues = Object.entries(queues).filter(([, v]) => v > 0);
        if (nonZeroQueues.length > 0) {
            html += '<div class="obs-section-title">Processing Queues</div>';
            html += '<div class="obs-stats">';
            for (const [name, depth] of nonZeroQueues) {
                html += obsStatCard(name.replace(/-/g, ' '), depth, 'items waiting');
            }
            html += '</div>';
        }

        el.innerHTML = html;
    } catch (e) {
        el.innerHTML = '<div class="obs-empty">Could not load memory data.</div>';
    }
}

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
            el.innerHTML = '<p class="obs-summary">No tools have been used yet.</p><div class="obs-empty">Tool performance data will appear here after tools are invoked.</div>';
            return;
        }

        let html = `<p class="obs-summary">Chalie has used ${tools.length} tool${tools.length === 1 ? '' : 's'} in the last 30 days.</p>`;

        // Success rate bar chart
        html += '<div class="obs-section-title">Success Rate</div>';
        html += '<div class="obs-bar-chart">';
        for (const t of tools) {
            const pct = obsPct(t.success_rate);
            const colorClass = pct >= 90 ? '--success' : pct >= 70 ? '--warning' : '--error';
            html += `<div class="obs-bar-row">
                <span class="obs-bar-row__label">${escapeHtml(t.tool_name)}</span>
                <div class="obs-bar-row__track"><div class="obs-bar-row__fill ${colorClass}" style="width:${pct}%"></div></div>
                <span class="obs-bar-row__value">${pct}%</span>
            </div>`;
        }
        html += '</div>';

        // Per-tool detail cards
        html += '<div class="obs-section-title">Details</div>';
        for (const t of tools) {
            const lastUsed = t.last_used_at ? timeAgo(t.last_used_at) : 'unknown';
            html += `<div class="obs-task-card">
                <div class="obs-task-card__header">
                    <span class="obs-task-card__title">${escapeHtml(t.tool_name)}</span>
                    <span class="obs-task-card__badge --active">${t.total} invocations</span>
                </div>
                <div class="obs-task-card__meta">
                    Avg latency: ${t.avg_latency}ms · Last used: ${escapeHtml(lastUsed)}
                </div>
            </div>`;
        }

        el.innerHTML = html;
    } catch (e) {
        el.innerHTML = '<div class="obs-empty">Could not load tool data.</div>';
    }
}

// ── Identity ──

async function loadIdentityObs() {
    const el = document.getElementById('identityContent');
    el.innerHTML = obsSkeletonBlock(40) + obsSkeletonBlock(200);

    try {
        const res = await apiFetch('/system/observability/identity');
        if (!res.ok) throw new Error('Failed to load');
        const data = await res.json();
        obsData.identity = data;
        obsLoaded.identity = true;
        obsSetTimestamp(data.generated_at);

        const vectors = data.vectors || {};
        const names = Object.keys(vectors);

        if (names.length === 0) {
            el.innerHTML = '<p class="obs-summary">Identity vectors have not been initialized yet.</p><div class="obs-empty">Identity data will appear here after interactions begin shaping personality.</div>';
            return;
        }

        let html = `<p class="obs-summary">Chalie's personality is expressed across ${names.length} dimensions, each shaped by interactions.</p>`;

        for (const name of names) {
            const v = vectors[name];
            const activation = v.activation || 0;
            const baseline = v.baseline || 0;
            const maxVal = v.max || 1;
            const activationPct = Math.min(100, (activation / maxVal) * 100);
            const baselinePct = Math.min(100, (baseline / maxVal) * 100);

            html += `<div class="obs-vector-row">
                <span class="obs-vector-row__label">${escapeHtml(name)}</span>
                <div class="obs-vector-row__track">
                    <div class="obs-vector-row__fill" style="width:${activationPct.toFixed(1)}%"></div>
                    <div class="obs-vector-row__baseline" style="left:${baselinePct.toFixed(1)}%"></div>
                </div>
                <span class="obs-vector-row__values">${activation.toFixed(2)} / ${baseline.toFixed(2)}</span>
            </div>`;
        }

        el.innerHTML = html;
    } catch (e) {
        el.innerHTML = '<div class="obs-empty">Could not load identity data.</div>';
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
        const threads = data.curiosity_threads || [];
        const cal = data.calibration || {};

        let html = `<p class="obs-summary">Chalie is currently working on ${tasks.length} background task${tasks.length === 1 ? '' : 's'} and exploring ${threads.length} curiosity thread${threads.length === 1 ? '' : 's'}.</p>`;

        // Triage calibration (framed as "Decision accuracy")
        if (cal.correct_rate !== undefined || cal.act_success_rate !== undefined) {
            html += '<div class="obs-section-title">Decision Accuracy</div>';
            html += '<div class="obs-stats">';
            if (cal.correct_rate !== undefined) {
                html += obsStatCard('Correct Rate', obsPct(cal.correct_rate) + '%', 'Routing decisions');
            }
            if (cal.act_success_rate !== undefined) {
                html += obsStatCard('Action Success', obsPct(cal.act_success_rate) + '%', 'Completed actions');
            }
            html += '</div>';
        }

        // Persistent tasks
        if (tasks.length > 0) {
            html += '<div class="obs-section-title">Background Tasks</div>';
            const statusLabels = { proposed: 'Proposed', accepted: 'Accepted', in_progress: 'In Progress', paused: 'Paused' };
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
                </div>`;
            }
        } else {
            html += '<div class="obs-section-title">Background Tasks</div>';
            html += '<div class="obs-empty">No active background tasks.</div>';
        }

        // Curiosity threads
        if (threads.length > 0) {
            html += '<div class="obs-section-title">Curiosity Threads</div>';
            for (const t of threads) {
                const typeClass = t.thread_type === 'learning' ? '--learning' : '--behavioral';
                const engagement = t.engagement_score !== undefined ? obsPct(t.engagement_score) + '% engagement' : '';
                const explorations = t.exploration_count !== undefined ? t.exploration_count + ' explorations' : '';
                const meta = [engagement, explorations].filter(Boolean).join(' · ');
                html += `<div class="obs-task-card">
                    <div class="obs-task-card__header">
                        <span class="obs-task-card__title">${escapeHtml(t.title || 'Untitled thread')}</span>
                        <span class="obs-task-card__badge ${typeClass}">${escapeHtml(t.thread_type || 'learning')}</span>
                    </div>
                    <div class="obs-task-card__meta">${escapeHtml(meta)}</div>
                </div>`;
            }
        } else {
            html += '<div class="obs-section-title">Curiosity Threads</div>';
            html += '<div class="obs-empty">No active curiosity threads.</div>';
        }

        el.innerHTML = html;
    } catch (e) {
        el.innerHTML = '<div class="obs-empty">Could not load task data.</div>';
    }
}

// ── Understanding (Autobiography + Traits) ──

const TRAIT_CATEGORY_LABELS = {
    core: 'About You',
    communication_style: 'How You Communicate',
    relationship: 'Our Relationship',
    preference: 'Your Preferences',
    physical: 'Physical',
    general: 'General',
    micro_preference: 'Small Preferences',
};

async function loadUnderstandingObs() {
    const el = document.getElementById('understandingContent');
    el.innerHTML = obsSkeletonBlock(60) + obsSkeletonBlock(120) + obsSkeletonBlock(80);

    try {
        const [autoRes, traitsRes] = await Promise.all([
            apiFetch('/system/observability/autobiography'),
            apiFetch('/system/observability/traits'),
        ]);

        const autoData = autoRes.ok ? await autoRes.json() : {};
        const traitsData = traitsRes.ok ? await traitsRes.json() : {};

        obsData.understanding = { autobiography: autoData, traits: traitsData };
        obsLoaded.understanding = true;
        obsSetTimestamp(autoData.generated_at || traitsData.generated_at);

        let html = '';

        // ── Autobiography section ──
        html += '<div class="obs-section-title">Autobiography';
        if (autoData.created_at) {
            html += ` <span class="obs-understanding-updated">Last updated ${timeAgo(autoData.created_at)}</span>`;
        }
        html += '</div>';

        if (autoData.narrative) {
            const changedSections = (autoData.delta && autoData.delta.changed) || [];
            const sections = autoData.narrative.split(/(?=^## )/m);

            for (const section of sections) {
                const trimmed = section.trim();
                if (!trimmed) continue;

                const headerMatch = trimmed.match(/^## (.+)/);
                const title = headerMatch ? headerMatch[1] : 'Overview';
                const body = headerMatch ? trimmed.slice(headerMatch[0].length).trim() : trimmed;
                const isChanged = changedSections.some(c => c.toLowerCase() === title.toLowerCase());

                html += `<div class="obs-understanding-section${isChanged ? ' --changed' : ''}">
                    <button class="obs-understanding-section__header" onclick="this.parentElement.classList.toggle('--open')">
                        <span>${escapeHtml(title)}</span>
                        ${isChanged ? '<span class="obs-understanding-section__delta">updated</span>' : ''}
                        <svg class="obs-understanding-section__chevron" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"></polyline></svg>
                    </button>
                    <div class="obs-understanding-section__body">${escapeHtml(body).replace(/\n/g, '<br>')}</div>
                </div>`;
            }
        } else {
            html += '<div class="obs-empty">No autobiography yet. Chalie needs a few more conversations to build your narrative.</div>';
        }

        // ── Traits section ──
        html += '<div class="obs-section-title">What I\'ve Learned About You</div>';

        const categories = traitsData.categories || {};
        const catKeys = Object.keys(categories);

        if (catKeys.length > 0) {
            for (const cat of catKeys) {
                const label = TRAIT_CATEGORY_LABELS[cat] || cat.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
                const traits = categories[cat];

                html += `<div class="obs-trait-category">
                    <div class="obs-trait-category__label">${escapeHtml(label)}</div>`;

                for (const t of traits) {
                    const conf = Math.round((t.confidence || 0) * 100);
                    const confClass = conf >= 70 ? '--high' : conf >= 40 ? '--mid' : '--low';
                    const keyLabel = escapeHtml((t.key || '').replace(/_/g, ' '));
                    const reinforcements = t.reinforcement_count || 0;

                    html += `<div class="obs-trait-item">
                        <div class="obs-trait-item__content">
                            <span class="obs-trait-item__key">${keyLabel}</span>
                            <span class="obs-trait-item__value">${escapeHtml(t.value || '')}</span>
                        </div>
                        <span class="obs-trait-item__confidence ${confClass}" title="${conf}% confidence">${conf}%</span>
                        <span class="obs-trait-item__reinforcements" title="${reinforcements} reinforcements">${reinforcements}x</span>
                        <button class="obs-trait-item__delete" data-trait-key="${escapeHtml(t.key)}" title="Remove this">×</button>
                    </div>`;
                }

                html += '</div>';
            }
        } else {
            html += '<div class="obs-empty">No traits learned yet. Chalie picks these up naturally from conversations.</div>';
        }

        el.innerHTML = html;

        // Wire up delete buttons
        el.querySelectorAll('.obs-trait-item__delete').forEach(btn => {
            btn.addEventListener('click', async (e) => {
                const key = e.target.dataset.traitKey;
                if (!key) return;
                if (!confirm(`Remove "${key.replace(/_/g, ' ')}"?`)) return;

                try {
                    const res = await apiFetch(`/system/observability/traits/${encodeURIComponent(key)}`, { method: 'DELETE' });
                    if (res.ok) {
                        showToast("Got it — I'll adjust.");
                        e.target.closest('.obs-trait-item').remove();
                    } else {
                        showToast('Could not remove trait.');
                    }
                } catch {
                    showToast('Could not remove trait.');
                }
            });
        });
    } catch (e) {
        el.innerHTML = '<div class="obs-empty">Could not load understanding data.</div>';
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
    } catch { return 'unknown'; }
}

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
        // Multi-word = full class string supplied (e.g. "fa-brands fa-google")
        // Single word = solid icon shorthand (e.g. "fa-cloud")
        const classes = icon.includes(' ') ? escapeHtml(icon) : `fa-solid ${escapeHtml(icon)}`;
        return `<i class="${classes}"></i>`;
    }
    return escapeHtml(icon);
}

// ==========================================
// Start
// ==========================================
init();
