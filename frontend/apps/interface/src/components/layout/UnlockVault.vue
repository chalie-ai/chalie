<!-- Vault-locked chokepoint. Polls GET /auth/status; while vault_state==='locked'
     it blocks the UI and submits POST /auth/login (stored login username +
     typed password) to unseal. Dismisses itself once vault_state==='unlocked'. -->
<script setup lang="ts">
import { ref, onMounted, onBeforeUnmount } from 'vue';
import { getUsername } from '@chalie/shared';
import { system } from '../../api/system';
import { auth } from '../../api/auth';

const locked = ref<boolean>(false);
const password = ref<string>('');
const error = ref<string>('');
const submitting = ref<boolean>(false);
let _poll: ReturnType<typeof setInterval> | null = null;

async function refreshState(): Promise<void> {
  try {
    locked.value = (await system.authStatus()).vault_state === 'locked';
  } catch {
    // Network hiccup — leave the current locked state; the poll retries.
  }
}

async function submit(): Promise<void> {
  if (submitting.value || !password.value) return;
  submitting.value = true;
  error.value = '';
  try {
    const result = await auth.login(getUsername(), password.value);
    if (result.vault_state === 'unlocked') {
      locked.value = false;
      password.value = '';
      if (_poll) {
        clearInterval(_poll);
        _poll = null;
      }
      globalThis.location.replace('/');
    } else {
      error.value = result.error ?? 'Could not unlock the vault.';
    }
  } catch {
    error.value = 'Invalid password.';
  } finally {
    submitting.value = false;
  }
}

onMounted(() => {
  void refreshState();
  _poll = setInterval(() => void refreshState(), 5000);
});
onBeforeUnmount(() => {
  if (_poll) clearInterval(_poll);
});
</script>

<template>
  <div v-if="locked" class="unlock-vault" role="dialog" aria-modal="true" aria-label="Unlock vault">
    <form class="unlock-vault__card" @submit.prevent="submit">
      <h1 class="unlock-vault__title">Unlock Chalie</h1>
      <p class="unlock-vault__hint">Enter your password to open the encrypted vault.</p>
      <input
        v-model="password"
        type="password"
        class="unlock-vault__input"
        autocomplete="current-password"
        placeholder="Password"
        aria-label="Password"
      />
      <button type="submit" class="unlock-vault__submit" :disabled="submitting || !password">
        {{ submitting ? 'Unlocking…' : 'Unlock' }}
      </button>
      <p v-if="error" class="unlock-vault__error" role="alert">{{ error }}</p>
    </form>
  </div>
</template>

<style scoped lang="scss">
.unlock-vault {
  position: fixed;
  inset: 0;
  z-index: 1000;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--bg-base);
}
.unlock-vault__card {
  display: flex;
  flex-direction: column;
  gap: var(--space-md);
  width: min(90vw, 24rem);
  padding: var(--space-xl);
  border-radius: var(--radius-lg);
  background: var(--bg-surface-2);
  border: 1px solid var(--border-subtle);
  text-align: center;
}
.unlock-vault__title { font-size: 1.35rem; font-weight: 600; color: var(--text-primary); }
.unlock-vault__hint { color: var(--text-secondary); line-height: 1.5; }
.unlock-vault__input {
  padding: var(--space-sm) var(--space-md);
  border-radius: var(--radius-md);
  border: 1px solid var(--border-subtle);
  background: var(--bg-input);
  color: var(--text-primary);
}
.unlock-vault__submit {
  padding: var(--space-sm);
  border-radius: var(--radius-full);
  border: none;
  background: var(--accent-primary);
  color: #fff;
  font-weight: 600;
  cursor: pointer;
  &:disabled { opacity: 0.6; cursor: default; }
}
.unlock-vault__error { color: var(--danger); min-height: 1.25rem; }
</style>
