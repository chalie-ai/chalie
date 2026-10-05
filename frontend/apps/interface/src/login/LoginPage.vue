<script setup lang="ts">
import { onMounted, ref } from 'vue';
import { auth, AuthError, HttpError } from '../api/auth';

// Only honour ?next= paths starting with '/' to prevent open-redirect.
function getNextDest(): string {
  const next = new URLSearchParams(window.location.search).get('next');
  return next && next.startsWith('/') ? next : '/';
}

const username = ref('');
const password = ref('');
const pending = ref(false);
const errorMsg = ref('');

const usernameInput = ref<HTMLInputElement | null>(null);

onMounted(() => {
  usernameInput.value?.focus();
});

async function handleSubmit() {
  errorMsg.value = '';

  if (!username.value.trim() || !password.value) {
    errorMsg.value = 'Username and password required.';
    return;
  }

  pending.value = true;
  try {
    await auth.login(username.value.trim(), password.value);
    window.location.replace(getNextDest());
  } catch (err) {
    if (err instanceof AuthError) {
      errorMsg.value = 'Invalid credentials.';
    } else if (err instanceof HttpError) {
      errorMsg.value = 'Login failed.';
    } else {
      errorMsg.value = 'Network error.';
    }
    pending.value = false;
  }
}
</script>

<template>
  <div class="login-card">
    <h1>Welcome back</h1>
    <p>Sign in to continue to Chalie.</p>

    <form @submit.prevent="handleSubmit">
      <label for="username">Username</label>
      <input
        id="username"
        ref="usernameInput"
        v-model="username"
        type="text"
        name="username"
        autocomplete="username"
        required
        :disabled="pending"
      />

      <label for="password">Password</label>
      <input
        id="password"
        v-model="password"
        type="password"
        name="password"
        autocomplete="current-password"
        required
        :disabled="pending"
      />

      <button type="submit" :disabled="pending">
        {{ pending ? 'Signing in...' : 'Sign in' }}
      </button>

      <div class="login-error">{{ errorMsg }}</div>
    </form>
  </div>
</template>

<style scoped lang="scss">
.login-card {
  width: 100%;
  max-width: 420px;
  padding: 2.5rem 2rem;
  background: var(--surface);
  border: 1px solid var(--line);

  h1 {
    font-size: 1.4rem;
    font-weight: 500;
    margin-bottom: 0.25rem;
  }

  p {
    color: var(--muted);
    font-size: 0.85rem;
    margin-bottom: 1.5rem;
  }

  label {
    display: block;
    font-size: 0.8rem;
    color: var(--muted);
    margin-bottom: 0.35rem;
  }

  input {
    display: block;
    width: 100%;
    padding: 0.6rem 0.75rem;
    font-size: 0.9rem;
    color: var(--text);
    background: var(--surface-2);
    border: 1px solid var(--control);
    margin-bottom: 1rem;
    outline: none;
    transition: border-color var(--dur-1) var(--ease-out);
    box-sizing: border-box;

    &:focus {
      border-color: var(--pink-text);
    }

    &:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }
  }

  button[type='submit'] {
    width: 100%;
    padding: 0.65rem;
    font-size: 0.9rem;
    font-weight: 500;
    color: var(--on-pink);
    background: var(--pink);
    border: none;
    cursor: pointer;
    transition:
      background-color var(--dur-1) var(--ease-out),
      color var(--dur-1) var(--ease-out),
      translate var(--dur-1) var(--ease-out);

    &:hover:not(:disabled) {
      background: var(--text);
      color: var(--bg);
    }

    &:active:not(:disabled) {
      translate: 0 1px;
    }

    &:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }
  }
}

.login-error {
  color: var(--deny-text);
  font-size: 0.8rem;
  margin-top: 0.75rem;
  min-height: 1.2em;
}
</style>
