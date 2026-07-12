<script setup lang="ts">
// Whole-instance snapshot Time-Machine. Import stages a FULL WIPE restore and
// restarts the instance (not a merge).
import { ref } from 'vue';
import { snapshot } from '../api/snapshot';
import { useToast } from '../composables/useToast';
import { useConfirm } from '../composables/useConfirm';
import { apiErrorMessage } from '../api/http';
import { DatabaseBackup, Download, Upload } from '@lucide/vue';

const { show: showToast } = useToast();
const { confirm } = useConfirm();

const exportPass = ref('');
const importFile = ref<File | null>(null);
const importPass = ref('');

async function doExport(): Promise<void> {
  showToast('Exporting snapshot…', 'info');
  try {
    const res = await snapshot.export(exportPass.value || null);
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: 'Export failed' }));
      throw new Error((err as { error?: string }).error || 'Export failed');
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `chalie-snapshot-${new Date().toISOString().replace(/[:.]/g, '-')}.zip`;
    a.click();
    URL.revokeObjectURL(url);
    showToast('Snapshot exported', 'success');
  } catch (e) {
    showToast(e instanceof Error ? e.message : 'Export failed', 'error');
  }
}

function onFileChosen(e: Event): void {
  importFile.value = (e.target as HTMLInputElement).files?.[0] ?? null;
}

async function confirmImport(): Promise<void> {
  const file = importFile.value;
  if (!file) {
    showToast('Choose a snapshot file first', 'error');
    return;
  }
  const ok = await confirm({
    title: 'Restore from snapshot?',
    desc:
      `This will completely WIPE AND OVERRIDE ALL existing data with the contents of ` +
      `"${file.name}". This is a full restore, not a merge, and cannot be undone. ` +
      `The instance will restart to apply it.`,
    confirmLabel: 'Wipe & Restore',
    confirmClass: 'btn-danger',
  });
  if (!ok) return;

  const form = new FormData();
  form.append('file', file);
  if (importPass.value) form.append('password', importPass.value);

  showToast('Staging restore…', 'info');
  try {
    await snapshot.import(form);
    showToast('Restore staged — restarting to apply…', 'success', { duration: 8000 });
    setTimeout(() => location.reload(), 6000);
  } catch (e) {
    showToast(apiErrorMessage(e, 'Import failed'), 'error');
  }
}
</script>

<template>
  <div class="panel-header">
    <h2><DatabaseBackup :size="20" /> Import / Export</h2>
  </div>

  <div class="brain-overview">
    <div class="export-card">
      <div class="export-card-icon"><Download :size="24" /></div>
      <div class="export-card-label">Export a full snapshot</div>
      <p class="form-hint">
        Downloads a single .zip that is a complete clone of this instance — databases, documents,
        skills, and secrets. Set a password to encrypt it with AES-256.
      </p>
      <label class="form-label"
        >Password (optional)
        <input
          v-model="exportPass"
          type="password"
          class="form-input"
          autocomplete="new-password"
        />
      </label>
      <button class="btn btn-primary" @click="doExport">
        <Download :size="14" /> Export Snapshot
      </button>
    </div>

    <div class="export-card">
      <div class="export-card-icon"><Upload :size="24" /></div>
      <div class="export-card-label">Restore from a snapshot</div>
      <p class="form-hint">
        Importing a snapshot <strong>completely wipes and overrides ALL existing data</strong> in
        this instance. This is a full restore, not a merge. The instance restarts to apply it.
      </p>
      <label class="form-label"
        >Snapshot file
        <input type="file" class="form-input" accept=".zip" @change="onFileChosen" />
      </label>
      <label class="form-label"
        >Password (if encrypted)
        <input
          v-model="importPass"
          type="password"
          class="form-input"
          autocomplete="current-password"
        />
      </label>
      <button class="btn btn-danger" @click="confirmImport">
        <Upload :size="14" /> Import &amp; Restore
      </button>
    </div>
  </div>
</template>
