<script setup lang="ts">
import { ref, computed } from 'vue';
import { documents } from '../../api/documents';
import type { Document, WatchedFolder } from '../../api/documents';
import { formatDate } from '../../utils/format';
import { useToast } from '../../composables/useToast';
import { useConfirm } from '../../composables/useConfirm';
import { HttpError } from '@chalie/shared';
import { useBrainResource } from '../../composables/useBrainResource';
import { Upload, ChevronLeft, Eye, FileText, Trash2 } from '@lucide/vue';

const props = defineProps<{
  statusFilter: 'active' | 'processing' | 'uploads' | 'deleted';
}>();

const GROUP_TABS = [
  { key: 'all', label: 'All' },
  { key: 'doc_category', label: 'Category' },
  { key: 'doc_project', label: 'Project' },
  { key: 'doc_date', label: 'Date' },
] as const;

const { show: showToast } = useToast();
const { confirm } = useConfirm();

interface DocsPayload { docs: Document[]; folders: WatchedFolder[] }

const { data: docsPayload, loading, reload: load } = useBrainResource(
  async () => {
    const [docRes, wfRes] = await Promise.all([
      documents.list(props.statusFilter === 'deleted'),
      documents.watchedFolders(),
    ]);
    return { docs: docRes.items ?? [], folders: wfRes.items ?? [] } satisfies DocsPayload;
  },
  { initial: { docs: [], folders: [] } as DocsPayload, failMsg: 'Failed to load documents' },
);

const docs = computed(() => docsPayload.value.docs);
const folders = computed(() => docsPayload.value.folders);

const search = ref('');
const groupBy = ref('all');
const drillGroup = ref<string | null>(null);
const viewMode = ref<'list' | 'detail'>('list');
const detailDoc = ref<Document | null>(null);
const fileInput = ref<HTMLInputElement | null>(null);

const filtered = computed<Document[]>(() => {
  let result = docs.value;
  if (props.statusFilter === 'active') {
    result = result.filter((d) => (d.status as string) === 'ready');
  } else if (props.statusFilter === 'processing') {
    result = result.filter((d) =>
      ['building', 'processing', 'pending'].includes((d.status as string) ?? ''),
    );
  } else if (props.statusFilter === 'uploads') {
    result = result.filter((d) => (d.source_type as string) === 'upload');
  } else if (props.statusFilter === 'deleted') {
    result = result.filter((d) => (d.deleted_at as string | null) != null);
  }
  if (search.value) {
    const q = search.value.toLowerCase();
    result = result.filter(
      (d) =>
        ((d.original_name as string) || '').toLowerCase().includes(q) ||
        ((d.doc_category as string) || '').toLowerCase().includes(q),
    );
  }
  return result;
});

type GroupEntry = { name: string; count: number };

const groupEntries = computed<GroupEntry[]>(() => {
  if (groupBy.value === 'all') return [];
  const groups: Record<string, number> = {};
  for (const d of filtered.value) {
    const k = String(d[groupBy.value] ?? 'Other') || 'Other';
    groups[k] = (groups[k] ?? 0) + 1;
  }
  return Object.entries(groups)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([name, count]) => ({ name, count }));
});

const drillDocs = computed<Document[]>(() => {
  if (!drillGroup.value) return filtered.value;
  return filtered.value.filter((d) => (String(d[groupBy.value] ?? 'Other') || 'Other') === drillGroup.value);
});

function statusClass(status: unknown): string {
  const map: Record<string, string> = {
    ready: 'badge-success',
    building: 'badge-warning',
    processing: 'badge-warning',
    failed: 'badge-danger',
    error: 'badge-danger',
    deleted: 'badge-muted',
  };
  return map[(status as string) ?? ''] ?? 'badge-muted';
}

function fmtSize(bytes: unknown): string {
  const n = bytes as number | null | undefined;
  if (!n) return '—';
  if (n >= 1048576) return `${(n / 1048576).toFixed(1)} MB`;
  if (n >= 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${n} B`;
}

async function onSearch(event: Event): Promise<void> {
  search.value = (event.target as HTMLInputElement).value;
  drillGroup.value = null;
  await load();
}

async function viewDoc(d: Document): Promise<void> {
  try {
    detailDoc.value = (await documents.get(d.id)).item;
    viewMode.value = 'detail';
  } catch (e) {
    showToast(e instanceof HttpError ? 'Failed to load document' : 'Network error', 'error');
  }
}

async function deleteDoc(d: Document): Promise<void> {
  const ok = await confirm({
    title: 'Delete Document',
    desc: `Delete "${(d.original_name as string) || 'this document'}"? This cannot be undone.`,
    confirmLabel: 'Delete',
    confirmClass: 'btn-danger',
  });
  if (!ok) return;
  try {
    await documents.delete(d.id);
    showToast('Document deleted', 'success');
    drillGroup.value = null;
    await load();
  } catch (e) {
    showToast(e instanceof HttpError ? 'Delete failed' : 'Network error', 'error');
  }
}

async function onUpload(event: Event): Promise<void> {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  if (!file) return;
  const fd = new FormData();
  fd.append('file', file);
  try {
    await documents.upload(fd);
    showToast('Document uploaded', 'success');
    drillGroup.value = null;
    await load();
  } catch (e) {
    showToast(e instanceof HttpError ? (e.error || 'Upload failed') : 'Network error', 'error');
  } finally {
    input.value = '';
  }
}
</script>

<template>
  <div class="panel-header">
    <h2>Documents</h2>
    <div class="panel-header-actions">
      <input
        type="text"
        class="search-input"
        placeholder="Search documents…"
        @keydown.enter="onSearch"
      >
      <button class="btn btn-primary" @click="fileInput?.click()">
        <Upload :size="14" /> Upload
      </button>
    </div>
  </div>

  <input
    ref="fileInput"
    type="file"
    accept=".pdf,.docx,.md,.txt,.csv,.zip"
    style="display:none"
    @change="onUpload"
  >

  <template v-if="viewMode === 'detail'">
    <div class="provider-form-page" style="max-width:none">
      <div class="form-page-header">
        <button class="btn btn-secondary btn-sm back-btn" @click="viewMode = 'list'">
          <ChevronLeft :size="14" /> Back
        </button>
        <h3>{{ (detailDoc?.original_name as string) || 'Document' }}</h3>
      </div>
      <div class="doc-meta-row">
        <span v-if="detailDoc?.source_type" class="badge badge-muted">{{ detailDoc.source_type }}</span>
        <span v-if="detailDoc?.doc_category" class="badge badge-violet">{{ detailDoc.doc_category }}</span>
        <span v-if="detailDoc?.file_size_bytes" class="doc-meta-item">{{ fmtSize(detailDoc.file_size_bytes) }}</span>
        <span v-if="detailDoc?.created_at" class="doc-meta-item">{{ formatDate(detailDoc.created_at as string) }}</span>
      </div>
      <pre class="doc-content">{{ (detailDoc?.summary as string) || 'No content available.' }}</pre>
    </div>
  </template>

  <template v-else>
    <div class="doc-group-tabs">
      <button
        v-for="tab in GROUP_TABS"
        :key="tab.key"
        class="group-tab"
        :class="{ active: groupBy === tab.key }"
        @click="groupBy = tab.key; drillGroup = null"
      >
        {{ tab.label }}
      </button>
    </div>

    <div v-if="!drillGroup && folders.length > 0" class="watched-folders-section">
      <h4 class="section-head">Watched Folders</h4>
      <div class="watched-folders-list">
        <div v-for="f in folders" :key="f.id" class="watched-folder-item">
          <span class="wf-icon"><Eye :size="14" /></span>
          <span class="wf-label">{{ f.label || f.folder_path || '' }}</span>
          <span class="wf-path">{{ f.folder_path || '' }}</span>
          <span class="wf-meta">{{ f.last_scan_files || 0 }} files · {{ formatDate(f.last_scan_at) }}</span>
          <span class="wf-status">
            <span class="status-dot" :class="{ active: f.enabled }"></span>
          </span>
        </div>
      </div>
    </div>

    <div v-if="loading" class="loading">Loading…</div>

    <!-- Empty state takes precedence over grouping when nothing matches -->
    <template v-else-if="filtered.length === 0">
      <div class="empty-state">
        <div class="empty-icon">
          <FileText :size="40" />
        </div>
        <h3>No documents</h3>
        <p>Upload or watch a folder to get started.</p>
      </div>
    </template>

    <template v-else-if="groupBy !== 'all' && !drillGroup">
      <div class="group-cards-grid">
        <button
          v-for="entry in groupEntries"
          :key="entry.name"
          class="group-card"
          @click="drillGroup = entry.name"
        >
          <div class="group-card-name">{{ entry.name }}</div>
          <div class="group-card-count">{{ entry.count }} document{{ entry.count === 1 ? '' : 's' }}</div>
        </button>
      </div>
    </template>

    <template v-else>
      <div v-if="drillGroup" class="form-page-header">
        <button class="btn btn-secondary btn-sm back-btn" @click="drillGroup = null">
          <ChevronLeft :size="14" /> Back
        </button>
        <h3>{{ drillGroup }}</h3>
        <span class="doc-meta-item">{{ drillDocs.length }} document{{ drillDocs.length === 1 ? '' : 's' }}</span>
      </div>

      <table class="records-table">
        <thead>
          <tr>
            <th>Name</th>
            <th>Source</th>
            <th>Status</th>
            <th>Size</th>
            <th>Added</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="d in drillDocs" :key="d.id">
            <td class="key-cell">{{ (d.original_name as string) || '' }}</td>
            <td><span class="badge badge-muted">{{ (d.source_type as string) || '' }}</span></td>
            <td><span class="badge" :class="statusClass(d.status)">{{ (d.status as string) || '' }}</span></td>
            <td>{{ fmtSize(d.file_size_bytes) }}</td>
            <td>{{ formatDate(d.created_at as string) }}</td>
            <td class="row-actions">
              <button class="btn btn-sm btn-secondary" @click="viewDoc(d)">View</button>
              <button class="btn btn-sm btn-danger" @click="deleteDoc(d)">
                <Trash2 :size="12" />
              </button>
            </td>
          </tr>
        </tbody>
      </table>
    </template>
  </template>
</template>
