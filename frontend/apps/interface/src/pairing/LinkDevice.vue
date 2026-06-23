<!-- Native pairing screen — scans the Brain-generated QR (PairingPayload),
     persists host + token + username, then reloads so the auth gate re-runs
     with the bearer attached. Reached only on the Tauri runtime with no token. -->
<script setup lang="ts">
import { ref } from 'vue';
import { setHost, setToken, setUsername, validatePairingPayload } from '@chalie/shared';
import { scan, Format } from '@tauri-apps/plugin-barcode-scanner';

const error = ref<string>('');
const scanning = ref<boolean>(false);

async function startScan(): Promise<void> {
  if (scanning.value) return;
  scanning.value = true;
  error.value = '';
  try {
    const result = await scan({ formats: [Format.QRCode], windowed: false });
    const payload = validatePairingPayload(JSON.parse(result.content));
    setHost(payload.host);
    setToken(payload.token);
    setUsername(payload.username); // so UnlockVault needs only a password.
    globalThis.location.replace('/');
  } catch (err: unknown) {
    error.value = err instanceof Error ? err.message : 'Pairing failed. Try again.';
  } finally {
    scanning.value = false;
  }
}
</script>

<template>
  <main class="link-device">
    <h1 class="link-device__title">Link this device</h1>
    <p class="link-device__hint">
      Open Chalie's Brain on your computer → Link device, then scan the QR code shown there.
    </p>
    <button class="link-device__scan" :disabled="scanning" @click="startScan">
      {{ scanning ? 'Scanning…' : 'Scan QR code' }}
    </button>
    <p v-if="error" class="link-device__error" role="alert">{{ error }}</p>
  </main>
</template>

<style scoped lang="scss">
.link-device {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: var(--space-lg);
  padding: var(--space-xl);
  background: var(--bg-base);
  color: var(--text-primary);
  text-align: center;
}
.link-device__title { font-size: 1.5rem; font-weight: 700; }
.link-device__hint { max-width: 28rem; color: var(--text-secondary); line-height: 1.5; }
.link-device__scan {
  padding: var(--space-sm) var(--space-xl);
  border-radius: var(--radius-full);
  border: none;
  background: var(--accent-primary);
  color: #fff;
  font-weight: 600;
  cursor: pointer;
  &:disabled { opacity: 0.6; cursor: default; }
}
.link-device__error { color: var(--danger); min-height: 1.25rem; }
</style>
