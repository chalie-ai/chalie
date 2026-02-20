// ==========================================
// Configuration
// ==========================================
const API_BASE = '';

// ==========================================
// State
// ==========================================
let currentPlatform = 'ollama';

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
function showPhase(phaseId) {
    const phases = ['accountPhase', 'loginPhase', 'setupPhase', 'completionPhase'];
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

    // Update model input placeholder
    const modelInput = document.getElementById('setupModel');
    modelInput.placeholder = config.modelPlaceholder;

    // Update datalist
    const datalist = document.getElementById('modelSuggestions');
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

// ==========================================
// Ollama Connection Test
// ==========================================
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
// Event Listeners
// ==========================================
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

    // Test connection button
    document.getElementById('testConnectionBtn').addEventListener('click', async () => {
        const models = await testOllamaConnection('setupHost', 'connectionStatus');
        if (models.length > 0) {
            const datalist = document.getElementById('modelSuggestions');
            datalist.innerHTML = '';
            models.forEach(m => {
                const opt = document.createElement('option');
                opt.value = m.name || m.model || m;
                datalist.appendChild(opt);
            });
        }
    });

    // Setup form submit
    document.getElementById('setupForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        await submitSetupForm();
    });

    // Completion buttons
    document.getElementById('gotoChalieBtn').addEventListener('click', () => {
        window.location.href = '/';
    });

    document.getElementById('gotoDashboardBtn').addEventListener('click', () => {
        window.location.href = '/brain/';
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

async function submitSetupForm() {
    const btn = document.getElementById('setupSubmitBtn');
    btn.disabled = true;
    btn.textContent = 'Saving...';

    const platform = currentPlatform;
    const config = PLATFORM_CONFIG[platform];

    const body = {
        name: document.getElementById('setupName').value.trim(),
        platform: platform,
        model: document.getElementById('setupModel').value.trim(),
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
            // Show completion screen
            document.getElementById('setupPhase').style.display = 'none';
            document.getElementById('completionPhase').style.display = '';

            showToast('Provider configured successfully!', 'success');
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
// Start
// ==========================================
init();
