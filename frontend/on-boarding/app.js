// ==========================================
// Configuration
// ==========================================
const API_BASE = '';

// ==========================================
// State
// ==========================================
let currentPlatform = 'ollama';
let ollamaSelectedModel = null;

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
        modelPlaceholder: 'e.g. claude-sonnet-4-20250514',
        models: ['claude-sonnet-4-20250514', 'claude-haiku-4-5-20251001'],
    },
    openai: {
        desc: 'API key from <a href="https://platform.openai.com/api-keys" target="_blank">platform.openai.com/api-keys</a>',
        hasHost: false,
        hasApiKey: true,
        modelPlaceholder: 'e.g. gpt-4o',
        models: ['gpt-4o', 'gpt-4o-mini', 'gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano', 'o3-mini'],
    },
    gemini: {
        desc: 'Free tier available — API key from <a href="https://aistudio.google.com/apikey" target="_blank">aistudio.google.com/apikey</a>',
        hasHost: false,
        hasApiKey: true,
        modelPlaceholder: 'e.g. gemini-2.5-flash',
        models: ['gemini-2.5-flash', 'gemini-2.5-pro'],
    },
};

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
// API Helpers
// ==========================================
async function apiFetch(path, options = {}) {
    const url = API_BASE ? `${API_BASE.replace(/\/$/, '')}${path}` : path;
    const headers = {
        'Content-Type': 'application/json',
        ...(options.headers || {}),
    };
    const response = await fetch(url, { ...options, headers, credentials: 'same-origin' });
    return response;
}

// ==========================================
// Phase Management
// ==========================================

/**
 * Show the named onboarding phase panel and hide all others.
 *
 * The `completionPhase` is no longer part of the flow — successful provider
 * setup redirects directly to `/`. Only the three active phases are toggled.
 *
 * @param {string} phaseId - One of 'accountPhase' | 'loginPhase' | 'setupPhase'.
 */
function showPhase(phaseId) {
    const phases = ['accountPhase', 'loginPhase', 'setupPhase', 'capabilitiesPhase'];
    phases.forEach(phase => {
        const el = document.getElementById(phase);
        if (el) {
            el.style.display = phase === phaseId ? '' : 'none';
        }
    });
}

// ==========================================
// Initialization
// ==========================================
async function init() {
    try {
        const r = await fetch('/auth/status', { credentials: 'same-origin' });
        if (r.ok) {
            const data = await r.json();
            const { has_master_account, has_providers, has_session } = data;

            if (has_master_account && has_session && has_providers) {
                // Fully set up and logged in, redirect home
                window.location.replace('/');
                return;
            }

            if (!has_master_account) {
                // Fresh install, show account creation
                showPhase('accountPhase');
            } else if (!has_session) {
                // Account exists but user not logged in
                showPhase('loginPhase');
            } else {
                // Logged in but no providers configured
                showPhase('setupPhase');
            }
        } else {
            // Error checking status, allow user to continue
            showPhase('accountPhase');
        }
    } catch (e) {
        // Backend unreachable, assume fresh install
        showPhase('accountPhase');
    }

    setupPlatformTabs();
    selectPlatform('ollama');
    setupEventListeners();
}

// ==========================================
// Platform Selection
// ==========================================
function setupPlatformTabs() {
    const tabs = document.querySelectorAll('.platform-tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const platform = tab.dataset.platform;
            selectPlatform(platform);
        });
    });
}

function selectPlatform(platform) {
    currentPlatform = platform;
    const config = PLATFORM_CONFIG[platform];

    // Update active tab
    document.querySelectorAll('.platform-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.platform === platform);
    });

    // Update description
    document.getElementById('platformDesc').innerHTML = config.desc;

    // Show/hide host field
    document.getElementById('setupHostGroup').style.display = config.hasHost ? '' : 'none';

    // Show/hide api key field
    document.getElementById('setupApiKeyGroup').style.display = config.hasApiKey ? '' : 'none';

    // Ollama: show model picker panel, hide free-text model group and test-connection row
    const ollamaGroup = document.getElementById('ollamaModelGroup');
    const textModelGroup = document.getElementById('setupModelGroup');
    const testConnRow = document.getElementById('testConnectionRow');
    if (platform === 'ollama') {
        if (ollamaGroup) ollamaGroup.style.display = '';
        if (textModelGroup) textModelGroup.style.display = 'none';
        if (testConnRow) testConnRow.style.display = 'none';
        // Reset selection state when switching to Ollama
        ollamaSelectedModel = null;
    } else {
        if (ollamaGroup) ollamaGroup.style.display = 'none';
        if (textModelGroup) textModelGroup.style.display = '';
        if (testConnRow) testConnRow.style.display = '';

        // Update model input placeholder and datalist for non-Ollama platforms
        const modelInput = document.getElementById('setupModel');
        if (modelInput) modelInput.placeholder = config.modelPlaceholder;

        const datalist = document.getElementById('modelSuggestions');
        if (datalist) {
            datalist.innerHTML = '';
            config.models.forEach(m => {
                const opt = document.createElement('option');
                opt.value = m;
                datalist.appendChild(opt);
            });
        }
    }

    // For Anthropic: fetch models when api key is entered
    if (platform === 'anthropic') {
        const apiKeyInput = document.getElementById('setupApiKey');
        apiKeyInput.oninput = debounce(() => {
            if (apiKeyInput.value.length > 20) {
                fetchAnthropicModels(apiKeyInput.value, 'modelSuggestions');
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
        const res = await apiFetch('/providers/anthropic/models', {
            method: 'POST',
            body: JSON.stringify({ api_key: key }),
        });
        if (res.ok) {
            const data = await res.json();
            const datalist = document.getElementById(datalistId);
            if (datalist) {
                datalist.innerHTML = '';
                (data.models || []).forEach(id => {
                    const opt = document.createElement('option');
                    opt.value = id;
                    datalist.appendChild(opt);
                });
            }
        } else {
            console.warn('[providers] Anthropic model fetch failed:', res.status);
        }
    } catch (e) {
        console.warn('[providers] Anthropic model fetch error:', e);
    }
}

// ==========================================
// Ollama Model Fetch & Picker
// ==========================================
async function refreshOllamaModels() {
    const statusEl = document.getElementById('ollamaRefreshStatus');
    if (statusEl) {
        statusEl.textContent = 'Fetching models…';
        statusEl.className = '';
    }

    const host = (document.getElementById('setupHost') || {}).value || 'http://localhost:11434';

    try {
        const res = await apiFetch(`/providers/ollama/models?host=${encodeURIComponent(host.trim() || 'http://localhost:11434')}`);
        if (res.ok) {
            const data = await res.json();
            const names = data.models || [];
            if (statusEl) {
                statusEl.textContent = names.length > 0 ? `${names.length} model(s) available` : 'Connected — no models installed yet';
                statusEl.className = 'status-ok';
            }
            renderOllamaModelPicker(names);
        } else {
            const err = await res.json().catch(() => ({}));
            if (statusEl) {
                statusEl.textContent = `✗ ${err.error || 'Connection failed'}`;
                statusEl.className = 'status-err';
            }
            renderOllamaModelPicker([]);
        }
    } catch (e) {
        if (statusEl) {
            statusEl.textContent = '✗ Cannot reach backend';
            statusEl.className = 'status-err';
        }
        renderOllamaModelPicker([]);
    }
}

function renderOllamaModelPicker(names) {
    const container = document.getElementById('ollamaModelPicker');
    if (!container) return;

    // Keep previously selected model if it still appears in the new list
    if (ollamaSelectedModel && !names.includes(ollamaSelectedModel)) {
        ollamaSelectedModel = null;
    }

    if (names.length === 0) {
        container.innerHTML = '<div class="ollama-empty-state">No models found. Is Ollama running on this host?<br><button type="button" class="btn-retry" id="ollamaRetryBtn">Retry</button></div>';
        const retryBtn = document.getElementById('ollamaRetryBtn');
        if (retryBtn) retryBtn.addEventListener('click', refreshOllamaModels);
        return;
    }

    container.innerHTML = names.map(n => {
        const sel = n === ollamaSelectedModel;
        return `<button type="button" class="ollama-model-chip${sel ? ' selected' : ''}" data-model="${n.replace(/"/g, '&quot;')}">${n.replace(/</g, '&lt;')}</button>`;
    }).join('');

    container.querySelectorAll('.ollama-model-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            ollamaSelectedModel = chip.dataset.model;
            container.querySelectorAll('.ollama-model-chip').forEach(c => {
                c.classList.toggle('selected', c.dataset.model === ollamaSelectedModel);
            });
        });
    });
}

// ==========================================
// Connection Test (non-Ollama providers via backend)
// ==========================================
async function testProviderConnection() {
    const statusEl = document.getElementById('connectionStatus');
    statusEl.textContent = 'Testing...';
    statusEl.className = '';

    const platform = currentPlatform;
    const config = PLATFORM_CONFIG[platform];
    const body = { platform };

    const model = document.getElementById('setupModel').value.trim();
    if (model) body.model = model;

    if (config.hasApiKey) {
        const key = document.getElementById('setupApiKey').value.trim();
        if (key) body.api_key = key;
    }

    // Quick client-side validation
    if (!body.model) {
        statusEl.textContent = '✗ Enter a model name first';
        statusEl.className = 'status-err';
        return;
    }
    if (config.hasApiKey && !body.api_key) {
        statusEl.textContent = '✗ Enter an API key first';
        statusEl.className = 'status-err';
        return;
    }

    try {
        const res = await apiFetch('/providers/test', {
            method: 'POST',
            body: JSON.stringify(body),
        });

        if (res.ok) {
            const data = await res.json();
            if (data.success) {
                statusEl.textContent = `✓ ${data.message || 'Connected'}`;
                statusEl.className = 'status-ok';
            } else {
                statusEl.textContent = `✗ ${data.error || 'Test failed'}`;
                statusEl.className = 'status-err';
            }
        } else {
            statusEl.textContent = '✗ Test request failed';
            statusEl.className = 'status-err';
        }
    } catch (e) {
        statusEl.textContent = '✗ Network error — cannot reach backend';
        statusEl.className = 'status-err';
    }
}

// ==========================================
// Event Listeners
// ==========================================

/**
 * Attach all interactive event listeners for the onboarding flow.
 *
 * Covers: account creation form, login form, provider connection test, and
 * the provider setup form. The former completion-phase buttons
 * (gotoChalieBtn / gotoDashboardBtn) have been removed; successful provider
 * setup now redirects directly to `/` inside {@link submitSetupForm}.
 */
function setupEventListeners() {
    // Account form submit
    document.getElementById('accountForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        await submitAccountForm();
    });

    // Login form submit
    document.getElementById('loginForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        await submitLoginForm();
    });

    // Ollama: refresh models button (single code path)
    let _ollamaRefreshInFlight = false;
    const _safeRefreshOllamaModels = async () => {
        if (_ollamaRefreshInFlight) return;
        _ollamaRefreshInFlight = true;
        try { await refreshOllamaModels(); } finally { _ollamaRefreshInFlight = false; }
    };
    document.getElementById('ollamaRefreshBtn').addEventListener('click', _safeRefreshOllamaModels);

    // Host input: clear stale picker state as soon as the value changes,
    // so a model selection from the previous host can't silently survive.
    document.getElementById('setupHost').addEventListener('input', () => {
        if (currentPlatform !== 'ollama') return;
        ollamaSelectedModel = null;
        renderOllamaModelPicker([]);
    });

    // Ollama: host blur → trigger refresh (guarded against double-fire when
    // the blur was caused by clicking the Refresh button itself).
    document.getElementById('setupHost').addEventListener('blur', () => {
        if (currentPlatform === 'ollama') _safeRefreshOllamaModels();
    });

    // Test connection button (non-Ollama platforms only)
    document.getElementById('testConnectionBtn').addEventListener('click', () => {
        testProviderConnection();
    });

    // Setup form submit
    document.getElementById('setupForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        await submitSetupForm();
    });
}

// ==========================================
// Form Submission
// ==========================================
async function submitAccountForm() {
    const username = document.getElementById('accountUsername').value.trim();
    const password = document.getElementById('accountPassword').value.trim();
    const confirmPassword = document.getElementById('accountConfirmPassword').value.trim();

    // Client-side validation
    if (!username) {
        showToast('Username required', 'error');
        return;
    }
    if (password.length < 8) {
        showToast('Password must be at least 8 characters', 'error');
        return;
    }
    if (password !== confirmPassword) {
        showToast('Passwords do not match', 'error');
        return;
    }

    const btn = document.querySelector('#accountForm button[type="submit"]');
    btn.disabled = true;
    btn.textContent = 'Creating...';

    try {
        const res = await fetch(API_BASE ? `${API_BASE.replace(/\/$/, '')}/auth/register` : '/auth/register', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password }),
        });

        if (res.ok) {
            // Session cookie is set by server
            showPhase('setupPhase');
            showToast('Account created successfully!', 'success');
        } else if (res.status === 409) {
            showToast('Account already exists', 'error');
        } else {
            const err = await res.json();
            showToast(err.error || 'Failed to create account', 'error');
        }
    } catch (e) {
        showToast('Network error', 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = 'Create Account';
    }
}

async function submitLoginForm() {
    const username = document.getElementById('loginUsername').value.trim();
    const password = document.getElementById('loginPassword').value.trim();

    if (!username || !password) {
        showToast('Username and password required', 'error');
        return;
    }

    const btn = document.querySelector('#loginForm button[type="submit"]');
    btn.disabled = true;
    btn.textContent = 'Logging in...';

    try {
        const res = await fetch(API_BASE ? `${API_BASE.replace(/\/$/, '')}/auth/login` : '/auth/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password }),
        });

        if (res.ok) {
            // Session cookie is set by server — check if providers are already configured
            const statusRes = await fetch('/auth/status', { credentials: 'same-origin' });
            if (statusRes.ok) {
                const status = await statusRes.json();
                if (status.has_providers) {
                    window.location.replace('/');
                    return;
                }
            }
            showPhase('setupPhase');
            showToast('Logged in successfully!', 'success');
        } else if (res.status === 401) {
            showToast('Invalid credentials', 'error');
        } else {
            const err = await res.json();
            showToast(err.error || 'Failed to login', 'error');
        }
    } catch (e) {
        showToast('Network error', 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = 'Login';
    }
}

/**
 * Submit the provider setup form to the backend.
 *
 * On success, redirects immediately to `/` — no completion screen is shown.
 * On error, re-enables the submit button so the user can correct and retry.
 *
 * @returns {Promise<void>}
 */
async function submitSetupForm() {
    const btn = document.getElementById('setupSubmitBtn');
    btn.disabled = true;
    btn.textContent = 'Saving...';

    const platform = currentPlatform;
    const config = PLATFORM_CONFIG[platform];

    // For Ollama, model comes from the picker; for others from the text input.
    let modelVal;
    if (platform === 'ollama') {
        modelVal = ollamaSelectedModel || '';
        if (!modelVal) {
            showToast('Select a model from the list first', 'error');
            btn.disabled = false;
            btn.textContent = 'Save Provider';
            return;
        }
    } else {
        modelVal = document.getElementById('setupModel').value.trim();
    }

    const body = {
        name: document.getElementById('setupName').value.trim(),
        platform: platform,
        model: modelVal,
        models: [modelVal],
    };

    if (config.hasHost) {
        body.host = document.getElementById('setupHost').value.trim() || 'http://localhost:11434';
    }
    if (config.hasApiKey) {
        const key = document.getElementById('setupApiKey').value.trim();
        if (key) body.api_key = key;
    }

    try {
        const res = await apiFetch('/providers', {
            method: 'POST',
            body: JSON.stringify(body),
        });

        if (res.ok) {
            // Provider saved — proceed to capability connection screen.
            showToast('Provider configured successfully!', 'success');
            showPhase('capabilitiesPhase');
            setupCapabilityListeners();
        } else {
            const err = await res.json();
            showToast(err.error || 'Failed to save provider', 'error');
            btn.disabled = false;
            btn.textContent = 'Save Provider';
        }
    } catch (e) {
        showToast('Network error', 'error');
        btn.disabled = false;
        btn.textContent = 'Save Provider';
    }
}

// ==========================================
// Capability Connection (Phase 4)
// ==========================================
let _capListenersAttached = false;

function setupCapabilityListeners() {
    if (_capListenersAttached) return;
    _capListenersAttached = true;

    const emailInput = document.getElementById('capEmail');
    emailInput.addEventListener('blur', () => autodiscoverProvider(emailInput.value.trim()));

    document.getElementById('capForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        await connectCapabilities();
    });

    document.getElementById('capSkipBtn').addEventListener('click', () => {
        window.location.href = '/';
    });
}

const EMAIL_PROVIDERS = {
    'gmail.com': { name: 'Google', note: 'Enable 2FA then create an App Password at myaccount.google.com/apppasswords' },
    'googlemail.com': { name: 'Google', note: 'Enable 2FA then create an App Password at myaccount.google.com/apppasswords' },
    'icloud.com': { name: 'Apple', note: 'Generate an app-specific password at appleid.apple.com' },
    'me.com': { name: 'Apple', note: 'Generate an app-specific password at appleid.apple.com' },
    'mac.com': { name: 'Apple', note: 'Generate an app-specific password at appleid.apple.com' },
    'outlook.com': { name: 'Outlook', note: 'Enable 2FA then create an App Password in Microsoft account security' },
    'hotmail.com': { name: 'Outlook', note: 'Enable 2FA then create an App Password in Microsoft account security' },
    'live.com': { name: 'Outlook', note: 'Enable 2FA then create an App Password in Microsoft account security' },
    'yahoo.com': { name: 'Yahoo', note: 'Generate an App Password at login.yahoo.com > Account Security' },
    'yahoo.co.uk': { name: 'Yahoo', note: 'Generate an App Password at login.yahoo.com > Account Security' },
};

function autodiscoverProvider(email) {
    const infoEl = document.getElementById('capProviderInfo');
    if (!email || !email.includes('@')) {
        infoEl.style.display = 'none';
        return;
    }
    const domain = email.split('@')[1].trim().toLowerCase();
    const provider = EMAIL_PROVIDERS[domain];
    if (provider) {
        let html = `<strong>${provider.name}</strong> detected.`;
        if (provider.note) html += ` <span class="provider-note">${provider.note}</span>`;
        infoEl.innerHTML = html;
        infoEl.style.display = '';
    } else {
        infoEl.innerHTML = 'Provider not recognised. Supported: Google, Apple, Outlook, Yahoo.';
        infoEl.style.display = '';
    }
}

async function connectCapabilities() {
    const email = document.getElementById('capEmail').value.trim();
    const password = document.getElementById('capPassword').value.trim();
    const btn = document.getElementById('capConnectBtn');
    const progressEl = document.getElementById('capProgress');
    const errorEl = document.getElementById('capError');
    const stepImap = document.getElementById('capStepImap');
    const stepCaldav = document.getElementById('capStepCaldav');

    if (!email || !password) {
        showToast('Email and app password required', 'error');
        return;
    }

    btn.disabled = true;
    btn.textContent = 'Connecting...';
    errorEl.style.display = 'none';
    progressEl.style.display = '';
    stepImap.className = 'cap-step active';
    stepImap.textContent = 'Connecting...';
    stepCaldav.className = 'cap-step';
    stepCaldav.textContent = '';

    try {
        const res = await apiFetch('/api/capabilities/mail/setup', {
            method: 'POST',
            body: JSON.stringify({ email, password }),
        });
        if (res.ok) {
            const data = await res.json();
            stepImap.className = 'cap-step done';
            stepImap.textContent = 'Connected ✓';
            // Show per-protocol status if available
            const statusRes = await apiFetch('/api/capabilities/mail/status');
            if (statusRes.ok) {
                const status = await statusRes.json();
                const protos = status.health_details?.protocols || {};
                const parts = [];
                if (protos.imap) parts.push('Email');
                if (protos.caldav) parts.push('Calendar');
                if (protos.carddav) parts.push('Contacts');
                if (parts.length) {
                    stepCaldav.className = 'cap-step done';
                    stepCaldav.textContent = parts.join(', ') + ' active';
                }
            }
            showToast('Connected! Redirecting...', 'success');
            setTimeout(() => { window.location.href = '/'; }, 1500);
        } else {
            const err = await res.json();
            stepImap.className = 'cap-step failed';
            stepImap.textContent = err.error || 'Connection failed';
            errorEl.textContent = 'Could not connect. Check your app password and try again, or skip for now.';
            errorEl.style.display = '';
            btn.disabled = false;
            btn.textContent = 'Retry';
        }
    } catch {
        stepImap.className = 'cap-step failed';
        stepImap.textContent = 'Network error';
        errorEl.textContent = 'Network error — please try again.';
        errorEl.style.display = '';
        btn.disabled = false;
        btn.textContent = 'Retry';
    }
}

// ==========================================
// Start
// ==========================================
init();

if (typeof lucide !== 'undefined') lucide.createIcons();
