<script setup lang="ts">
import { ref } from 'vue';
import { auth, HttpError } from '../api/auth';

type Phase = 'account' | 'voice';

const phase = ref<Phase>('account');

const username = ref('');
const password = ref('');
const confirmPassword = ref('');
const pending = ref(false);

interface Toast {
  id: number;
  message: string;
  type: 'success' | 'error' | 'info';
}
let nextId = 0;
const toasts = ref<Toast[]>([]);

// Drop from the queue after the 3s visible window; the TransitionGroup leave
// transition plays the 0.3s fade-out.
const TOAST_VISIBLE_MS = 3000;

function showToast(message: string, type: Toast['type'] = 'info'): void {
  const id = ++nextId;
  toasts.value.push({ id, message, type });
  setTimeout(() => {
    toasts.value = toasts.value.filter((t) => t.id !== id);
  }, TOAST_VISIBLE_MS);
}

async function handleAccountSubmit(): Promise<void> {
  if (!username.value.trim()) {
    showToast('Username required', 'error');
    return;
  }
  if (password.value.length < 8) {
    showToast('Password must be at least 8 characters', 'error');
    return;
  }
  if (password.value !== confirmPassword.value) {
    showToast('Passwords do not match', 'error');
    return;
  }

  pending.value = true;
  try {
    await auth.register(username.value.trim(), password.value);
    phase.value = 'voice';
  } catch (err) {
    if (err instanceof HttpError && err.status === 409) {
      showToast('Account already exists', 'error');
    } else if (err instanceof HttpError) {
      // Surface the server's {error} message when present, else a generic fallback.
      showToast(err.error ?? `Failed to create account (HTTP ${err.status})`, 'error');
    } else {
      showToast('Network error', 'error');
    }
  } finally {
    pending.value = false;
  }
}

async function enableVoice(): Promise<void> {
  try {
    await auth.setVoiceEnabled(true);
  } catch (e) {
    console.warn('[on-boarding] voice enable request failed:', e);
  }
  window.location.replace('/brain/');
}

function skipVoice(): void {
  window.location.replace('/brain/');
}
</script>

<template>
  <div class="ob-container">
    <div v-if="phase === 'account'" class="ob-card">
      <div class="ob-card-header">
        <h1>Create Master Account</h1>
        <p>Your gateway to the Chalie dashboard.</p>
      </div>

      <div class="warning-box">
        <h3>⚠ Critical — Read Before Continuing</h3>
        <p>This is your Master Account. It is the only credential that controls access to the Chalie dashboard and system configuration.</p>
        <p>There is no password recovery. If you forget your username or password, you will be permanently locked out. All cognitive memory and configuration will be inaccessible.</p>
        <p>Write your credentials down and store them somewhere safe before continuing.</p>
      </div>

      <form @submit.prevent="handleAccountSubmit">
        <div class="form-group">
          <label for="accountUsername">Username</label>
          <input
            id="accountUsername"
            v-model="username"
            type="text"
            autocomplete="username"
            required
            :disabled="pending"
          />
        </div>
        <div class="form-group">
          <label for="accountPassword">Password</label>
          <input
            id="accountPassword"
            v-model="password"
            type="password"
            required
            :disabled="pending"
          />
        </div>
        <div class="form-group">
          <label for="accountConfirmPassword">Confirm Password</label>
          <input
            id="accountConfirmPassword"
            v-model="confirmPassword"
            type="password"
            required
            :disabled="pending"
          />
        </div>
        <div class="form-actions">
          <button type="submit" class="btn-primary" :disabled="pending">
            {{ pending ? 'Creating...' : 'Create Account' }}
          </button>
        </div>
      </form>
    </div>

    <div v-else-if="phase === 'voice'" class="ob-card">
      <div class="ob-card-header">
        <h1>Voice Support</h1>
        <p>Chalie can speak responses and listen to voice input.</p>
      </div>

      <div class="info-box">
        <p>Enabling voice will download additional dependencies (~500 MB) in the background. Your GPU will be detected automatically for optimal performance.</p>
        <p>You can change this later in the Brain dashboard settings.</p>
      </div>

      <div class="form-actions">
        <button class="btn-primary" @click="enableVoice">Enable Voice</button>
        <button class="btn-secondary" @click="skipVoice">Skip</button>
      </div>
    </div>
  </div>

  <TransitionGroup tag="div" name="toast" class="toast-container">
    <div
      v-for="toast in toasts"
      :key="toast.id"
      class="toast"
      :class="`toast-${toast.type}`"
    >{{ toast.message }}</div>
  </TransitionGroup>
</template>

<style scoped lang="scss">
:global(body::before) {
  content: '';
  position: fixed;
  inset: 0;
  z-index: -1;
  pointer-events: none;
  background:
    radial-gradient(600px circle at 30% 20%, color-mix(in srgb, var(--accent-primary) 6%, transparent), transparent 60%),
    radial-gradient(500px circle at 70% 75%, color-mix(in srgb, var(--accent-tertiary) 4%, transparent), transparent 60%);
  animation: ambient-breathe 20s ease-in-out infinite;
}

@keyframes ambient-breathe {
  0%, 100% { transform: scale(1); opacity: 0.8; }
  50%       { transform: scale(1.06); opacity: 1; }
}

.ob-container {
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
}

.ob-card {
  width: 100%;
  max-width: 500px;
  padding: 40px;
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: var(--bs-border-radius-lg);
}

.ob-card-header {
  text-align: center;
  margin-bottom: 32px;

  h1 {
    font-size: 1.4rem;
    font-weight: 500;
    margin-bottom: 0.25rem;
  }

  p {
    color: var(--text-secondary);
    font-size: 0.85rem;
    margin-bottom: 0;
  }
}

.warning-box,
.info-box {
  border-radius: 6px;
  padding: 16px;
  margin-bottom: 24px;

  p {
    font-size: 13px;
    color: var(--text);
    line-height: 1.6;
    margin-bottom: 6px;

    &:last-child { margin-bottom: 0; }
  }
}

.warning-box {
  background: color-mix(in srgb, var(--error) 8%, transparent);
  border: 1px solid color-mix(in srgb, var(--error) 35%, transparent);
  border-left: 3px solid var(--error);

  h3 {
    color: var(--error);
    font-size: 13px;
    margin-bottom: 8px;
  }
}

.info-box {
  background: color-mix(in srgb, var(--accent-primary) 8%, transparent);
  border: 1px solid color-mix(in srgb, var(--accent-primary) 35%, transparent);
  border-left: 3px solid var(--accent-primary);
}

.form-group {
  margin-bottom: 20px;

  label {
    display: block;
    font-size: 0.8rem;
    color: var(--text-secondary);
    margin-bottom: 0.35rem;
  }

  input {
    display: block;
    width: 100%;
    padding: 0.6rem 0.75rem;
    font-size: 0.9rem;
    color: var(--text);
    background: var(--bg-input);
    border: 1px solid var(--border);
    border-radius: var(--bs-border-radius);
    outline: none;
    transition: border-color 0.2s;
    box-sizing: border-box;

    &:focus {
      border-color: color-mix(in srgb, var(--accent-primary) 35%, transparent);
      box-shadow: 0 0 8px color-mix(in srgb, var(--accent-primary) 6%, transparent);
    }

    &:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }
  }
}

.form-actions {
  display: flex;
  gap: 12px;
  margin-top: 28px;
}

.btn-primary,
.btn-secondary {
  flex: 1;
  padding: 0.65rem;
  font-size: 0.9rem;
  font-weight: 500;
  border-radius: var(--bs-border-radius);
  cursor: pointer;
  transition: opacity 0.2s;
}

.btn-primary {
  color: #fff;
  background: var(--accent-primary);
  border: none;

  &:hover:not(:disabled) { opacity: 0.85; }

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
}

.btn-secondary {
  color: var(--text);
  background: transparent;
  border: 1px solid var(--border);

  &:hover { opacity: 0.75; }
}

.toast-container {
  position: fixed;
  bottom: 20px;
  right: 20px;
  z-index: 10000;
  display: flex;
  flex-direction: column;
  gap: 10px;
  pointer-events: none;
}

.toast {
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 12px 16px;
  font-size: 13px;
  font-weight: 500;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
  pointer-events: auto;
  animation: slideIn 0.3s ease;

  &.toast-success { border-left: 3px solid var(--success); }
  &.toast-error   { border-left: 3px solid var(--error); }
  &.toast-info    { border-left: 3px solid var(--accent-primary); }
}

@keyframes slideIn {
  from { transform: translateX(400px); opacity: 0; }
  to   { transform: translateX(0);     opacity: 1; }
}

.toast-leave-active {
  transition: opacity 0.3s ease;
}
.toast-leave-to {
  opacity: 0;
}

@media (max-width: 600px) {
  .ob-card { padding: 24px; }

  .toast-container { left: 10px; right: 10px; bottom: 10px; }
  .toast { font-size: 12px; }
}
</style>
