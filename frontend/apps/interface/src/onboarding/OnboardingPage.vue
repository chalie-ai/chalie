<script setup lang="ts">
import { ref } from 'vue';
import { auth, HttpError } from '../api/auth';

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
// transition plays the fade-out.
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
    window.location.replace('/brain/');
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
</script>

<template>
  <div class="ob-container">
    <div class="ob-card">
      <div class="ob-card-header">
        <h1>Create Master Account</h1>
        <p>Your gateway to the Chalie dashboard.</p>
      </div>

      <div class="warning-box">
        <h3>⚠ Critical — Read Before Continuing</h3>
        <p>
          This is your Master Account. It is the only credential that controls access to the Chalie
          dashboard and system configuration.
        </p>
        <p>
          There is no password recovery. If you forget your username or password, you will be
          permanently locked out. All cognitive memory and configuration will be inaccessible.
        </p>
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
          <button type="submit" class="btn btn-primary" :disabled="pending">
            {{ pending ? 'Creating...' : 'Create Account' }}
          </button>
        </div>
      </form>
    </div>
  </div>

  <TransitionGroup tag="div" name="toast" class="toast-container">
    <div v-for="toast in toasts" :key="toast.id" class="toast" :class="`toast-${toast.type}`">
      {{ toast.message }}
    </div>
  </TransitionGroup>
</template>

<style scoped lang="scss">
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
  background: var(--surface);
}

.ob-card-header {
  text-align: center;
  margin-bottom: 32px;

  h1 {
    font-size: var(--fs-title);
    font-weight: 500;
    margin-bottom: 0.25rem;
  }

  p {
    color: var(--muted);
    font-size: var(--fs-body);
    margin-bottom: 0;
  }
}

.warning-box {
  padding: 16px;
  margin-bottom: 24px;
  background: var(--surface-2);

  h3 {
    color: var(--deny-text);
    font-size: var(--fs-body);
    margin-bottom: 8px;
  }

  p {
    font-size: var(--fs-body);
    color: var(--text);
    line-height: 1.6;
    margin-bottom: 6px;

    &:last-child {
      margin-bottom: 0;
    }
  }
}

.form-group {
  margin-bottom: 20px;

  label {
    display: block;
    font-size: var(--fs-body);
    color: var(--muted);
    margin-bottom: 0.35rem;
  }

  input {
    display: block;
    width: 100%;

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

.btn-primary {
  flex: 1;
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
  background: var(--surface-2);
  padding: 12px 16px;
  font-size: var(--fs-body);
  font-weight: 500;
  pointer-events: auto;
  animation: rise var(--dur-2) var(--ease-out);

  &.toast-success {
    color: var(--allow-text);
  }
  &.toast-error {
    color: var(--deny-text);
  }
  &.toast-info {
    color: var(--pink-text);
  }
}

.toast-leave-active {
  transition: opacity var(--dur-2) var(--ease-out);
}
.toast-leave-to {
  opacity: 0;
}

@media (max-width: 600px) {
  .ob-card {
    padding: 24px;
  }

  .toast-container {
    left: 10px;
    right: 10px;
    bottom: 10px;
  }
}
</style>
