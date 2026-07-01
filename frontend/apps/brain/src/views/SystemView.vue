<script setup lang="ts">
import { computed, onMounted, ref } from 'vue';
import { Lock, Network } from '@lucide/vue';
import ToggleSwitch from '../ui/ToggleSwitch.vue';
import { useToast } from '../composables/useToast';
import { useConfirm } from '../composables/useConfirm';
import type { NetworkConfig } from '../api/system';
import { system } from '../api/system';

const { show: showToast } = useToast();
const { confirm } = useConfirm();

const loading = ref(true);
const restarting = ref(false);

const domain = ref('');
const sslEnabled = ref(false);
const sslCertPresent = ref(false);
const sslCertFile = ref<File | null>(null);
const sslKeyFile = ref<File | null>(null);

const sslSaveBlocked = computed(
  () =>
    (sslEnabled.value && !sslCertPresent.value && !sslCertFile.value) ||
    !!sslCertFile.value !== !!sslKeyFile.value,
);

onMounted(async () => {
  try {
    const cfg: NetworkConfig = await system.getNetwork();
    domain.value = cfg.deployment_domain;
    sslEnabled.value = cfg.ssl_enabled;
    sslCertPresent.value = cfg.ssl_cert_present;
  } catch {
    showToast('Failed to load network settings', 'error');
  } finally {
    loading.value = false;
  }
});

function onCertChosen(e: Event): void {
  sslCertFile.value = (e.target as HTMLInputElement).files?.[0] ?? null;
}

function onKeyChosen(e: Event): void {
  sslKeyFile.value = (e.target as HTMLInputElement).files?.[0] ?? null;
}

async function save(): Promise<void> {
  const desc = sslEnabled.value
    ? 'Saving will restart Chalie (~15 s) and serve the site over https — after the restart, reconnect at https://<your-host>.'
    : 'Saving will restart Chalie (~15 s). The page will reconnect automatically.';

  const ok = await confirm({
    title: 'Restart required',
    desc,
    confirmLabel: 'Save & Restart',
    confirmClass: 'btn-primary',
  });
  if (!ok) return;

  restarting.value = true;
  try {
    const result = await system.saveNetwork({
      deployment_domain: domain.value.trim(),
      ssl_enabled: sslEnabled.value,
      ssl_cert: sslCertFile.value ?? undefined,
      ssl_key: sslKeyFile.value ?? undefined,
    });
    showToast('Chalie is restarting… reconnecting', 'success', { duration: 15000 });
    setTimeout(() => {
      if (result.ssl_enabled) {
        window.location.href = 'https://' + window.location.host + window.location.pathname;
      } else {
        window.location.reload();
      }
    }, 4000);
  } catch (e) {
    restarting.value = false;
    showToast(e instanceof Error ? e.message : 'Save failed', 'error');
  }
}
</script>

<template>
  <div class="panel-header">
    <h2><Network :size="20" /> System</h2>
  </div>

  <div v-if="loading" class="loading">Loading…</div>

  <div v-else-if="restarting" class="network-restart-banner">
    <Lock :size="18" />
    <span>Chalie is restarting… the page will reconnect automatically.</span>
  </div>

  <div v-else class="brain-overview">
    <!-- Domain card -->
    <div class="export-card">
      <div class="export-card-icon"><Network :size="24" /></div>
      <div class="export-card-label">Domain</div>
      <p class="form-hint">
        The public domain Chalie is served from. Used for CORS: leave blank to allow same-origin
        requests only, or set to your external domain (e.g. <code>chalie.example.com</code>) to
        allow cross-origin access.
      </p>
      <label class="form-label">
        Deployment domain
        <input
          v-model="domain"
          type="text"
          class="form-input"
          placeholder="e.g. chalie.example.com (blank = same-origin only)"
        />
      </label>
      <button class="btn btn-primary" @click="save"><Network :size="14" /> Save Domain</button>
    </div>

    <!-- SSL / TLS card -->
    <div class="export-card">
      <div class="export-card-icon"><Lock :size="24" /></div>
      <div class="export-card-label">SSL / TLS</div>
      <p class="form-hint">
        Enable HTTPS. Upload a PEM certificate and private key. After saving, Chalie restarts and
        the site switches to <code>https://</code> — you will need to reconnect at the new address.
      </p>

      <div class="network-toggle-row">
        <ToggleSwitch v-model="sslEnabled" label="Enable SSL" />
      </div>

      <div class="network-cert-status">
        <span v-if="sslCertPresent" class="badge badge-success">Certificate on file</span>
        <span v-else class="badge badge-muted">No certificate uploaded</span>
      </div>

      <label class="form-label">
        Certificate (PEM)
        <input type="file" class="form-input" accept=".pem,.crt,.cer" @change="onCertChosen" />
      </label>

      <label class="form-label">
        Private key (PEM)
        <input type="file" class="form-input" accept=".pem,.key" @change="onKeyChosen" />
      </label>

      <p v-if="sslSaveBlocked" class="network-ssl-warn">
        Upload both a certificate and a private key — or keep the existing certificate on file —
        before saving.
      </p>

      <button class="btn btn-primary" :disabled="sslSaveBlocked" @click="save">
        <Lock :size="14" /> Save SSL Settings
      </button>
    </div>
  </div>
</template>

<style scoped>
.network-toggle-row {
  display: flex;
  align-items: center;
  gap: 10px;
  margin: 12px 0 8px;
}

.network-cert-status {
  margin-bottom: 12px;
}

.network-ssl-warn {
  font-size: 12px;
  color: var(--error);
  background: rgba(var(--bs-danger-rgb), 0.1);
  border: 1px solid rgba(var(--bs-danger-rgb), 0.25);
  border-radius: 6px;
  padding: 8px 12px;
  margin: 8px 0;
  text-align: left;
}

.network-restart-banner {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 20px 24px;
  border-radius: 10px;
  background: rgba(var(--bs-warning-rgb), 0.1);
  border: 1px solid rgba(var(--bs-warning-rgb), 0.25);
  color: var(--warning-banner-text);
  font-size: 14px;
  font-weight: 500;
}
</style>
