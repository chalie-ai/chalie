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
// Error body reader — tolerates non-JSON responses
// ==========================================
async function readErrorMessage(res, fallback) {
    try {
        const text = await res.text();
        if (!text) return fallback + ' (HTTP ' + res.status + ')';
        try {
            const parsed = JSON.parse(text);
            return parsed.error || fallback + ' (HTTP ' + res.status + ')';
        } catch {
            return fallback + ' (HTTP ' + res.status + ')';
        }
    } catch {
        return fallback + ' (HTTP ' + res.status + ')';
    }
}

// ==========================================
// Phase Management
// ==========================================
function showPhase(phaseId) {
    ['accountPhase', 'loginPhase'].forEach(phase => {
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
            const { has_master_account, has_session } = data;

            if (!has_master_account) {
                showPhase('accountPhase');
            } else if (!has_session) {
                showPhase('loginPhase');
            } else {
                window.location.replace('/');
                return;
            }
        } else {
            showPhase('accountPhase');
        }
    } catch (e) {
        showPhase('accountPhase');
    }

    setupEventListeners();
}

// ==========================================
// Event Listeners
// ==========================================
function setupEventListeners() {
    document.getElementById('accountForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        await submitAccountForm();
    });

    document.getElementById('loginForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        await submitLoginForm();
    });
}

// ==========================================
// Form Submission
// ==========================================
async function submitAccountForm() {
    const username = document.getElementById('accountUsername').value.trim();
    const password = document.getElementById('accountPassword').value.trim();
    const confirmPassword = document.getElementById('accountConfirmPassword').value.trim();

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
        const res = await fetch('/auth/register', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password }),
        });

        if (res.ok) {
            window.location.replace('/');
        } else if (res.status === 409) {
            showToast('Account already exists', 'error');
            btn.disabled = false;
            btn.textContent = 'Create Account';
        } else {
            const msg = await readErrorMessage(res, 'Failed to create account');
            showToast(msg, 'error');
            btn.disabled = false;
            btn.textContent = 'Create Account';
        }
    } catch (e) {
        showToast('Network error', 'error');
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
        const res = await fetch('/auth/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password }),
        });

        if (res.ok) {
            window.location.replace('/');
        } else if (res.status === 401) {
            showToast('Invalid credentials', 'error');
            btn.disabled = false;
            btn.textContent = 'Login';
        } else {
            const msg = await readErrorMessage(res, 'Failed to login');
            showToast(msg, 'error');
            btn.disabled = false;
            btn.textContent = 'Login';
        }
    } catch (e) {
        showToast('Network error', 'error');
        btn.disabled = false;
        btn.textContent = 'Login';
    }
}

// ==========================================
// Start
// ==========================================
init();

if (typeof lucide !== 'undefined') lucide.createIcons();
