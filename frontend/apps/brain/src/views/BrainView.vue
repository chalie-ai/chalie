<script setup lang="ts">
import { ref, computed } from 'vue';
import { brain } from '../api/brain';
import type { BrainInfo } from '../api/brain';
import { useToast } from '../composables/useToast';
import { useConfirm } from '../composables/useConfirm';
import { useAsyncResource } from '@chalie/shared';
import BrainModal from '../ui/BrainModal.vue';
import BrainIcon from '../ui/BrainIcon.vue';
import { apiErrorMessage } from '../api/http';

const { show: showToast } = useToast();
const { confirm } = useConfirm();

// ── Info state ────────────────────────────────────────────────────────────────
const { data: info, loading, error: loadError } = useAsyncResource(
  () => brain.info(),
  { initial: null as BrainInfo | null },
);

// ── Export modal state ────────────────────────────────────────────────────────
const showExport = ref(false);
const expFiles = ref(true);
const expEncrypt = ref(true);
const expPass = ref('');
const expPassConfirm = ref('');

// ── Import state ──────────────────────────────────────────────────────────────
const fileInput = ref<HTMLInputElement | null>(null);
const importFile = ref<File | null>(null);
const showImportPass = ref(false);
const impPass = ref('');

// ── GitHub form state ─────────────────────────────────────────────────────────
const ghRepo = ref('');
const ghToken = ref('');
const ghStatus = ref('');

// ── GitHub upload modal state ─────────────────────────────────────────────────
const showGhUpload = ref(false);
const ghUploadPass = ref('');

// ── Computed ──────────────────────────────────────────────────────────────────
const dbSize = computed(() => info.value?.db_size_human || '0 B');
const docCount = computed(() => info.value?.document_count || 0);

// ── openExport (brain.js:89-107) ─────────────────────────────────────────────
function openExport(): void {
  expFiles.value = true;
  expEncrypt.value = true;
  expPass.value = '';
  expPassConfirm.value = '';
  showExport.value = true;
}

// ── doExport (brain.js:103-147) ──────────────────────────────────────────────
// The legacy modal closes before onConfirm fires — reproduce by closing first.
async function doExport(): Promise<void> {
  showExport.value = false;

  if (expEncrypt.value) {
    if (!expPass.value || expPass.value.length < 4) {
      showToast('Passphrase must be at least 4 characters', 'error');
      return;
    }
    if (expPass.value !== expPassConfirm.value) {
      showToast('Passphrases do not match', 'error');
      return;
    }
  }

  showToast('Exporting brain…', 'info');
  try {
    const res = await brain.export({
      include_files: expFiles.value,
      encrypt: expEncrypt.value,
      password: expEncrypt.value ? expPass.value : null,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: 'Export failed' }));
      throw new Error((err as { error?: string }).error || 'Export failed');
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `brain-export-${new Date().toISOString().replace(/[:.]/g, '-')}.chalie-backup`;
    a.click();
    URL.revokeObjectURL(url);
    showToast('Brain exported successfully', 'success');
  } catch (e) {
    showToast(e instanceof Error ? e.message : 'Export failed', 'error');
  }
}

// ── openImport (brain.js:149-164) ────────────────────────────────────────────
async function openImport(): Promise<void> {
  const ok = await confirm({
    title: 'Restore from Backup',
    desc: 'This will replace all current data. A pre-restore snapshot is saved automatically.',
    confirmLabel: 'Choose Backup File',
    confirmClass: 'btn-danger',
  });
  if (!ok) return;
  fileInput.value?.click();
}

// ── onFileChosen — triggered by the hidden <input type="file"> ────────────────
function onFileChosen(e: Event): void {
  const input = e.target as HTMLInputElement;
  const file = input.files?.[0];
  input.value = ''; // allow re-selecting the same file
  if (!file) return;
  importFile.value = file;
  impPass.value = '';
  showImportPass.value = true;
}

// ── doImport (brain.js:183-212) ──────────────────────────────────────────────
// Modal closes first (matches legacy _showModal confirm handler).
async function doImport(): Promise<void> {
  showImportPass.value = false;
  const file = importFile.value;
  if (!file) return;

  if (!impPass.value) {
    showToast('Passphrase is required', 'error');
    return;
  }

  showToast('Restoring brain…', 'info');
  const form = new FormData();
  form.append('file', file);
  form.append('password', impPass.value);

  try {
    const data = await brain.import(form);
    showToast(
      `Restored! DB: ${(data.db_size / 1024 / 1024).toFixed(1)} MB, Files: ${data.files_restored}`,
      'success',
      { duration: 6000 },
    );
    setTimeout(() => location.reload(), 2000);
  } catch (e) {
    showToast(apiErrorMessage(e, 'Import failed'), 'error');
  }
}

// ── testGithub (brain.js:214-232) ────────────────────────────────────────────
async function testGithub(): Promise<void> {
  const token = ghToken.value;
  if (!token) {
    showToast('Enter a GitHub token first', 'error');
    return;
  }
  ghStatus.value = 'Testing…';
  try {
    const data = await brain.testGithub({ github_token: token });
    ghStatus.value = `Connected as @${data.username}`;
    showToast(`GitHub connected as @${data.username}`, 'success');
  } catch (e) {
    ghStatus.value = 'Connection failed';
    showToast(apiErrorMessage(e, 'GitHub test failed'), 'error');
  }
}

// ── openGhUpload (brain.js:234-248) ──────────────────────────────────────────
function openGhUpload(): void {
  ghUploadPass.value = '';
  showGhUpload.value = true;
}

// ── doGhUpload (brain.js:250-285) ────────────────────────────────────────────
// Modal closes first (matches legacy _showModal confirm handler).
async function doGhUpload(): Promise<void> {
  showGhUpload.value = false;
  const repo = ghRepo.value.trim();
  const token = ghToken.value.trim();
  if (!repo || !token) {
    showToast('Enter repo and token', 'error');
    return;
  }

  const password = ghUploadPass.value;
  const encrypt = password.length >= 4;
  if (password && password.length < 4) {
    showToast('Passphrase must be at least 4 characters', 'error');
    return;
  }

  showToast('Exporting and uploading to GitHub…', 'info');
  try {
    const data = await brain.exportUpload({
      include_files: true,
      encrypt,
      password: encrypt ? password : null,
      github_repo: repo,
      github_token: token,
    });
    showToast('Backup uploaded to GitHub!', 'success', { duration: 5000 });
    if (data.commit_url) {
      ghStatus.value = 'Last upload: ' + new Date().toISOString().slice(0, 10);
    }
  } catch (e) {
    showToast(apiErrorMessage(e, 'Upload failed'), 'error');
  }
}

</script>

<template>
  <!-- Header (brain.js:18-20) -->
  <div class="panel-header">
    <h2><BrainIcon name="Brain" :size="20" /> Brain</h2>
  </div>

  <!-- Hidden file input for import (brain.js:156-162) -->
  <input
    ref="fileInput"
    type="file"
    accept=".chalie-backup"
    style="display: none"
    @change="onFileChosen"
  >

  <!-- Loading state (brain.js:22) -->
  <div v-if="loading" class="loading">Loading brain info…</div>

  <!-- Error / empty state (brain.js:34) -->
  <div v-else-if="loadError || !info" class="empty-state">
    <p>Could not load brain info.</p>
  </div>

  <!-- Brain overview (brain.js:46-81) -->
  <div v-else class="brain-overview">
    <!-- Stats cards (brain.js:47-58) -->
    <div class="brain-stats">
      <div class="export-card">
        <div class="export-card-icon"><BrainIcon name="Database" :size="24" /></div>
        <div class="export-card-label">Database</div>
        <div class="export-card-value">{{ dbSize }}</div>
      </div>
      <div class="export-card">
        <div class="export-card-icon"><BrainIcon name="Document" :size="24" /></div>
        <div class="export-card-label">Documents</div>
        <div class="export-card-value">{{ docCount }}</div>
      </div>
    </div>

    <!-- Action buttons (brain.js:60-63) -->
    <div class="brain-actions">
      <button class="btn btn-primary" @click="openExport">
        <BrainIcon name="Download" :size="14" /> Export Brain
      </button>
      <button class="btn btn-secondary" @click="openImport">
        <BrainIcon name="Upload" :size="14" /> Restore from Backup
      </button>
    </div>

    <!-- GitHub Backup section (brain.js:65-80) -->
    <div class="brain-section">
      <h3 class="brain-section-title">GitHub Backup</h3>
      <div class="brain-form-grid">
        <label class="form-label">Repository
          <input
            v-model="ghRepo"
            type="text"
            class="form-input"
            placeholder="owner/chalie-brain-backup"
          >
        </label>
        <label class="form-label">Token
          <input
            v-model="ghToken"
            type="password"
            class="form-input"
            placeholder="ghp_..."
          >
        </label>
      </div>
      <div class="brain-form-actions">
        <button class="btn btn-sm btn-secondary" @click="testGithub">Test Connection</button>
        <button class="btn btn-sm btn-primary" @click="openGhUpload">Upload Backup</button>
        <span class="form-hint">{{ ghStatus }}</span>
      </div>
    </div>
  </div>

  <!-- Export Brain modal (brain.js:89-107) -->
  <BrainModal v-model="showExport">
    <div class="modal-header">
      <h3>Export Brain</h3>
      <button class="btn-close" type="button" @click="showExport = false">
        <BrainIcon name="Close" :size="16" />
      </button>
    </div>
    <form @submit.prevent="doExport">
      <label class="form-label">
        <input v-model="expFiles" type="checkbox"> Include document files
      </label>
      <label class="form-label">
        <input v-model="expEncrypt" type="checkbox"> Encrypt with passphrase
      </label>
      <label class="form-label">Passphrase
        <input
          v-model="expPass"
          type="password"
          class="form-input"
          autocomplete="new-password"
        >
      </label>
      <label class="form-label">Confirm passphrase
        <input
          v-model="expPassConfirm"
          type="password"
          class="form-input"
          autocomplete="new-password"
        >
      </label>
      <div class="form-actions">
        <button type="button" class="btn btn-secondary" @click="showExport = false">Cancel</button>
        <button type="submit" class="btn btn-primary">Export</button>
      </div>
    </form>
  </BrainModal>

  <!-- Import passphrase modal (brain.js:167-181) -->
  <BrainModal v-model="showImportPass">
    <div class="modal-header">
      <h3>Enter Passphrase</h3>
      <button class="btn-close" type="button" @click="showImportPass = false">
        <BrainIcon name="Close" :size="16" />
      </button>
    </div>
    <form @submit.prevent="doImport">
      <p>Decrypting: <strong>{{ importFile?.name }}</strong></p>
      <label class="form-label">Passphrase
        <input
          v-model="impPass"
          type="password"
          class="form-input"
          autocomplete="current-password"
        >
      </label>
      <div class="form-actions">
        <button type="button" class="btn btn-secondary" @click="showImportPass = false">Cancel</button>
        <button type="submit" class="btn btn-danger">Restore</button>
      </div>
    </form>
  </BrainModal>

  <!-- GitHub upload modal (brain.js:234-248) -->
  <BrainModal v-model="showGhUpload">
    <div class="modal-header">
      <h3>Upload Backup to GitHub</h3>
      <button class="btn-close" type="button" @click="showGhUpload = false">
        <BrainIcon name="Close" :size="16" />
      </button>
    </div>
    <form @submit.prevent="doGhUpload">
      <p class="modal-desc">The backup will be encrypted and stored as a file in the repo.</p>
      <label class="form-label">Encryption passphrase (optional)
        <input
          v-model="ghUploadPass"
          type="password"
          class="form-input"
          autocomplete="new-password"
        >
      </label>
      <div class="form-actions">
        <button type="button" class="btn btn-secondary" @click="showGhUpload = false">Cancel</button>
        <button type="submit" class="btn btn-primary">Upload</button>
      </div>
    </form>
  </BrainModal>
</template>
