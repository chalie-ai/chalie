<script setup lang="ts">
import { ref, computed, onMounted, watch, onUnmounted } from 'vue';
import { providers } from '../api/providers';
import type { Provider, CatalogEntry } from '../api/providers';
import { apiErrorMessage } from '../api/http';
import { useToast } from '../composables/useToast';
import { useConfirm } from '../composables/useConfirm';
import { useShellStore } from '../stores/shell';
import BrainIcon from '../ui/BrainIcon.vue';

const { show: showToast } = useToast();
const { confirm } = useConfirm();
const shell = useShellStore();

// ── List state ────────────────────────────────────────────────────
const providerList = ref<Provider[]>([]);
const selectedId = ref<number | null>(null);
const loading = ref(true);

// ── Role selectors (vision + delegate, consolidated — TKT-960) ────
// source: 'explicit' (user-pinned) | 'auto' (fell back to main) | 'none'.
const visionId = ref<number | null>(null);
const visionSource = ref('none');
const delegateId = ref<number | null>(null);
const delegateSource = ref('none');

// ── Catalog cache ─────────────────────────────────────────────────
const catalog = ref<CatalogEntry[]>([]);
const catalogLoaded = ref(false);

// ── Wizard state ──────────────────────────────────────────────────
// 'list' | 'picker' | 'form'
const mode = ref<'list' | 'picker' | 'form'>('list');

const CUSTOM_PRESET: CatalogEntry = {
  id: 'custom',
  name: 'Custom (OpenAI-compatible)',
  platform: 'openai_compatible',
  host: '',
  needs_key: true,
};

const preset = ref<CatalogEntry | null>(null);
const editingId = ref<number | null>(null);
const editModel = ref('');

// Form fields
const formName = ref('');
const formHost = ref('');
const formKey = ref('');
const formModel = ref('');

// Model fetch state
const models = ref<string[]>([]);
const modelsFetchInFlight = ref(false);
const modelStatus = ref('');
const modelStatusClass = ref('');
const lastFetchKey = ref('');
let modelFetchTimer: ReturnType<typeof setTimeout> | null = null;

const MODEL_FETCH_DEBOUNCE_MS = 600;

// FIX 6: suppress the debounced watch that fires right after openWizard() pre-populates
// formHost/formKey and then directly calls fetchModels() itself. Without this, the watch
// would schedule a second debounced fetch on top of the one already in flight.
let suppressNextCredsWatch = false;

// Test button state
const testing = ref(false);

// ── Computed helpers ──────────────────────────────────────────────
const needsHost = computed(() =>
  preset.value?.platform === 'ollama' || preset.value?.platform === 'openai_compatible',
);

const needsKey = computed(() => !!preset.value?.needs_key);

const hostReady = computed(() => !needsHost.value || formHost.value.trim() !== '');

const keyReady = computed(() => !needsKey.value || formKey.value.trim() !== '');

const canFetch = computed(() => hostReady.value && keyReady.value);

const showHostGroup = computed(() => needsHost.value);

const showKeyGroup = computed(() => needsKey.value && hostReady.value);

const showModelGroup = computed(() => canFetch.value || models.value.length > 0);

const saveDisabled = computed(() => !formModel.value);

const isEditing = computed(() => editingId.value !== null);

// ── Catalog tiles ─────────────────────────────────────────────────
const catalogTiles = computed(() => [...catalog.value, CUSTOM_PRESET]);

function avatar(name: string): string {
  const s = (name || '').trim();
  return s ? s[0].toUpperCase() : '?';
}

// ── Mount ─────────────────────────────────────────────────────────
onMounted(async () => {
  await load();
});

onUnmounted(() => {
  if (modelFetchTimer !== null) {
    clearTimeout(modelFetchTimer);
    modelFetchTimer = null;
  }
});

// `loading` is initialized true (initial-mount placeholder, matching the
// legacy mount() innerHTML). Reloads after CRUD must NOT re-flash the
// placeholder — legacy _load() re-rendered the list in place — so this
// function never re-sets loading to true.
async function load(): Promise<void> {
  try {
    const [provRes, selRes, visRes, delRes] = await Promise.all([
      providers.list(),
      providers.getSelected(),
      providers.getVision(),
      providers.getDelegate(),
    ]);
    providerList.value = provRes.providers ?? [];
    selectedId.value = selRes.provider ? selRes.provider.id : null;
    visionId.value = visRes.provider ? visRes.provider.id : null;
    visionSource.value = visRes.source || 'none';
    delegateId.value = delRes.provider ? delRes.provider.id : null;
    delegateSource.value = delRes.source || 'none';
  } catch {
    showToast('Failed to connect to backend', 'error');
  }
  loading.value = false;
}

// ── Role selectors: options + current value ───────────────────────
const visionCapable = computed(() => providerList.value.filter((p) => p.supports_vision));

// "Use main provider" is only offered when the main provider itself can see.
const mainSupportsVision = computed(
  () => selectedId.value != null && visionCapable.value.some((p) => p.id === selectedId.value),
);

// Bound <select> value: explicit pin → its id, otherwise '' (use main / auto).
const visionSelectValue = computed(() =>
  visionSource.value === 'explicit' && visionId.value != null ? String(visionId.value) : '',
);

const delegateSelectValue = computed(() =>
  delegateSource.value === 'explicit' && delegateId.value != null ? String(delegateId.value) : '',
);

// value '' clears the explicit pin → fall back to the main provider.
async function setVision(value: string): Promise<void> {
  const pid = value === '' ? null : Number(value);
  try {
    const res = await providers.setVision(pid);
    visionId.value = res.provider ? res.provider.id : null;
    visionSource.value = res.source || (pid == null ? 'auto' : 'explicit');
    showToast('Vision provider updated', 'success');
  } catch (e: unknown) {
    showToast(apiErrorMessage(e, 'Failed', 'Failed to set vision provider'), 'error');
  }
}

async function setDelegate(value: string): Promise<void> {
  const pid = value === '' ? null : Number(value);
  try {
    const res = await providers.setDelegate(pid);
    delegateId.value = res.provider ? res.provider.id : null;
    delegateSource.value = res.source || (pid == null ? 'auto' : 'explicit');
    showToast('Delegate provider updated', 'success');
  } catch (e: unknown) {
    showToast(apiErrorMessage(e, 'Failed', 'Failed to set delegate provider'), 'error');
  }
}

// ── Select provider ───────────────────────────────────────────────
async function selectProvider(id: number): Promise<void> {
  try {
    await providers.setSelected(id);
    selectedId.value = id;
    showToast('Provider selected', 'success');
  } catch (e: unknown) {
    showToast(apiErrorMessage(e, 'Failed', 'Failed to select provider'), 'error');
  }
}

// ── Delete provider ───────────────────────────────────────────────
async function confirmDelete(id: number): Promise<void> {
  const p = providerList.value.find((x) => x.id === id);
  const name = p?.name || 'this provider';
  const confirmed = await confirm({
    title: 'Delete Provider',
    desc: `Delete "${name}"? This cannot be undone.`,
    confirmLabel: 'Delete',
    confirmClass: 'btn-danger',
  });
  if (!confirmed) return;
  try {
    await providers.delete(id);
    showToast('Provider deleted', 'success');
    await load();
  } catch (e: unknown) {
    showToast(apiErrorMessage(e, 'Delete failed'), 'error');
  }
}

// ── Wizard: open ─────────────────────────────────────────────────
async function openWizard(id: number | null): Promise<void> {
  editingId.value = id;
  editModel.value = '';
  models.value = [];
  lastFetchKey.value = '';
  modelStatus.value = '';
  modelStatusClass.value = '';
  if (modelFetchTimer !== null) {
    clearTimeout(modelFetchTimer);
    modelFetchTimer = null;
  }

  await ensureCatalog();

  if (id !== null) {
    const p = providerList.value.find((x) => x.id === id);
    if (!p) {
      mode.value = 'list';
      return;
    }
    editModel.value = p.model || '';
    models.value = editModel.value ? [editModel.value] : [];
    preset.value = presetFor(p);
    formName.value = p.name;
    // Suppress the creds watch: we mutate formHost/formKey here and call
    // fetchModels() directly below, so we must skip the debounced watch that
    // would otherwise schedule a redundant second fetch.
    suppressNextCredsWatch = true;
    formHost.value = p.host || '';
    formKey.value = '';
    formModel.value = editModel.value;
    mode.value = 'form';
    if (canFetch.value) {
      void fetchModels();
    }
  } else {
    preset.value = null;
    formName.value = '';
    formHost.value = '';
    formKey.value = '';
    formModel.value = '';
    mode.value = 'picker';
  }
}

function presetFor(provider: Provider): CatalogEntry {
  const match = catalog.value.find(
    (c) => c.platform === provider.platform && (c.host || '') === (provider.host || ''),
  );
  if (match) return match;
  return {
    id: 'edit',
    name: provider.name,
    platform: provider.platform,
    host: provider.host || '',
    needs_key: provider.platform !== 'ollama',
  };
}

async function ensureCatalog(): Promise<void> {
  if (catalogLoaded.value) return;
  try {
    const res = await providers.getCatalog();
    catalog.value = Array.isArray(res.catalog) ? res.catalog : [];
    catalogLoaded.value = true;  // only mark loaded on success so next open retries
  } catch {
    catalog.value = [];
    showToast('Could not load provider catalog', 'error');
    // catalogLoaded stays false → next wizard open will retry
  }
}

// ── Wizard: picker ────────────────────────────────────────────────
function selectTile(p: CatalogEntry): void {
  preset.value = { ...p };
  editModel.value = '';
  models.value = [];
  lastFetchKey.value = '';
  modelStatus.value = '';
  modelStatusClass.value = '';
  formName.value = p.name;
  // FIX 6 (cont.): like openWizard(), this mutates formHost/formKey and may call
  // fetchModels() directly below, so suppress the watch's redundant debounced fetch.
  suppressNextCredsWatch = true;
  formHost.value = p.host || '';
  formKey.value = '';
  formModel.value = '';
  mode.value = 'form';
  if (canFetch.value) {
    void fetchModels();
  }
}

function backFromPicker(): void {
  mode.value = 'list';
}

function backFromForm(): void {
  if (isEditing.value) {
    mode.value = 'list';
  } else {
    mode.value = 'picker';
  }
}

function cancelForm(): void {
  mode.value = 'list';
}

// ── Progressive reveal: watch creds inputs ────────────────────────
watch([() => formHost.value, () => formKey.value], () => {
  if (mode.value !== 'form') return;
  // FIX 6: openWizard() pre-populates these fields and calls fetchModels()
  // directly. Skip the debounced duplicate triggered by that mutation.
  if (suppressNextCredsWatch) {
    suppressNextCredsWatch = false;
    return;
  }
  debouncedFetchModels();
});

// ── Model fetch ───────────────────────────────────────────────────
function debouncedFetchModels(): void {
  if (modelFetchTimer !== null) {
    clearTimeout(modelFetchTimer);
  }
  modelFetchTimer = setTimeout(() => {
    if (canFetch.value) void fetchModels();
  }, MODEL_FETCH_DEBOUNCE_MS);
}

async function fetchModels(): Promise<void> {
  if (modelsFetchInFlight.value || !preset.value) return;

  const creds: { host?: string; api_key?: string } = {};
  if (needsHost.value) creds.host = formHost.value.trim();
  if (needsKey.value) creds.api_key = formKey.value.trim();

  const fetchKey = `${preset.value.platform}|${creds.host || ''}|${creds.api_key ? 'k' : ''}`;
  if (fetchKey === lastFetchKey.value && models.value.length > 0) return;
  lastFetchKey.value = fetchKey;

  modelsFetchInFlight.value = true;
  modelStatus.value = 'Loading models…';
  modelStatusClass.value = 'model-status loading';

  try {
    const data = await providers.listModels({ platform: preset.value.platform, ...creds });
    const fetched = (data.models ?? [])
      .map((m) => (typeof m === 'string' ? m : m?.id || ''))
      .filter(Boolean);
    models.value = fetched;

    // Preserve the editModel selection: append it as an extra option if not in the fetched list
    if (editModel.value && !models.value.includes(editModel.value)) {
      models.value = [...models.value, editModel.value];
    }

    // Restore the previously-selected model if it's in the new list
    if (editModel.value && models.value.includes(editModel.value)) {
      formModel.value = editModel.value;
    } else if (formModel.value && !models.value.includes(formModel.value)) {
      formModel.value = '';
    }

    const n = models.value.length;
    modelStatus.value = `${n} model${n === 1 ? '' : 's'}`;
    modelStatusClass.value = 'model-status ok';
  } catch (e: unknown) {
    modelStatus.value = apiErrorMessage(e, 'Failed to fetch models');
    modelStatusClass.value = 'model-status err';
    lastFetchKey.value = ''; // allow retry
  } finally {
    modelsFetchInFlight.value = false;
  }
}

// ── Test connection ───────────────────────────────────────────────
async function testConnection(): Promise<void> {
  if (!preset.value) return;
  testing.value = true;
  try {
    const body: {
      platform: string;
      name: string;
      model: string;
      host?: string;
      api_key?: string;
      provider_id?: number;
    } = {
      platform: preset.value.platform,
      name: formName.value.trim(),
      model: formModel.value,
    };
    if (needsHost.value) body.host = formHost.value.trim();
    if (needsKey.value) body.api_key = formKey.value.trim();
    if (editingId.value !== null) body.provider_id = editingId.value;

    const data = await providers.test(body);
    if (data.success) {
      showToast(data.message || 'Connection successful', 'success');
    } else {
      showToast(data.error || 'Test failed', 'error');
    }
  } catch (e: unknown) {
    showToast(apiErrorMessage(e, 'Test failed'), 'error');
  } finally {
    testing.value = false;
  }
}

// ── Save provider ─────────────────────────────────────────────────
async function saveProvider(): Promise<void> {
  if (!preset.value || !formModel.value) {
    showToast('Select a model first', 'error');
    return;
  }

  const body: {
    platform: string;
    name: string;
    model: string;
    host?: string;
    api_key?: string;
  } = {
    platform: preset.value.platform,
    name: formName.value.trim(),
    model: formModel.value,
  };
  if (needsHost.value) body.host = formHost.value.trim();
  if (needsKey.value) {
    const k = formKey.value.trim();
    if (k) body.api_key = k;
  }

  try {
    if (editingId.value !== null) {
      await providers.update(editingId.value, body);
    } else {
      await providers.create(body);
    }
    showToast(editingId.value !== null ? 'Provider updated' : 'Provider added', 'success');
    await load();
    mode.value = 'list';

    // Providers-only lift check
    if (shell.providersOnly) {
      try {
        const res = await fetch('/auth/status', { credentials: 'same-origin' });
        if (res.ok) {
          const sd = (await res.json()) as { has_providers?: boolean };
          if (sd.has_providers) shell.liftProvidersOnly();
        }
      } catch {
        // keep locked
      }
    }
  } catch (e: unknown) {
    showToast(apiErrorMessage(e, 'Save failed'), 'error');
  }
}
</script>

<template>
  <!-- ── List view ─────────────────────────────────────────────── -->
  <template v-if="mode === 'list'">
    <div class="panel-header">
      <h2>LLM Providers</h2>
      <button class="btn btn-primary" @click="openWizard(null)">
        <BrainIcon name="Plus" :size="14" />
        Add Provider
      </button>
    </div>

    <div class="providers-list">
      <template v-if="loading">
        <div class="loading">Loading providers…</div>
      </template>

      <template v-else-if="providerList.length === 0">
        <div class="empty-state">
          <div class="empty-icon">
            <BrainIcon name="Providers" :size="40" />
          </div>
          <h3>No providers</h3>
          <p>Add your first LLM provider to get started.</p>
        </div>
      </template>

      <template v-else>
        <div
          v-for="p in providerList"
          :key="p.id"
          :class="['provider-card', { selected: p.id === selectedId }]"
        >
          <label class="provider-radio">
            <input
              type="radio"
              name="sel_prov"
              :value="p.id"
              :checked="p.id === selectedId"
              @change="selectProvider(p.id)"
            />
            <span class="radio-dot"></span>
          </label>
          <div class="provider-info">
            <div class="provider-name">
              {{ p.name }}
              <span
                v-if="p.decrypt_failed"
                class="provider-warn"
                title="Misconfigured — re-enter credentials"
              >⚠</span>
            </div>
            <div class="provider-meta">
              <span :class="`badge badge-${p.platform}`">{{ p.platform }}</span>
              <span
                v-if="p.supports_vision"
                class="badge badge-success"
                title="Verified image understanding"
              >Vision</span>
              <span>{{ p.model || 'no model' }}</span>
              <span v-if="p.host">· {{ p.host }}</span>
            </div>
          </div>
          <div class="provider-actions">
            <button class="btn btn-sm btn-secondary" @click="openWizard(p.id)">Edit</button>
            <button class="btn btn-sm btn-danger" @click="confirmDelete(p.id)">Delete</button>
          </div>
        </div>
      </template>
    </div>

    <!-- ── Role selectors (vision + delegate) ─────────────────────── -->
    <template v-if="!loading && providerList.length > 0">
      <h4 class="section-head">Vision</h4>
      <p class="panel-desc">
        Choose which provider Chalie uses to understand images. Only providers that passed the
        vision probe can be selected.
      </p>
      <div class="form-group">
        <select
          v-if="visionCapable.length > 0"
          :value="visionSelectValue"
          @change="setVision(($event.target as HTMLSelectElement).value)"
        >
          <option v-if="mainSupportsVision" value="">Use main provider</option>
          <option v-for="p in visionCapable" :key="p.id" :value="p.id">{{ p.name }}</option>
        </select>
        <select v-else disabled>
          <option value="">Disabled — no vision-capable providers</option>
        </select>
      </div>

      <h4 class="section-head">Delegate</h4>
      <p class="panel-desc">
        Subagent work (web_search, web_browse, …) runs on this provider. Defaults to your main
        provider.
      </p>
      <div class="form-group">
        <select
          :value="delegateSelectValue"
          @change="setDelegate(($event.target as HTMLSelectElement).value)"
        >
          <option value="">Use main provider</option>
          <option v-for="p in providerList" :key="p.id" :value="p.id">{{ p.name }}</option>
        </select>
      </div>
    </template>
  </template>

  <!-- ── Picker ────────────────────────────────────────────────── -->
  <template v-else-if="mode === 'picker'">
    <div class="provider-wizard">
      <div class="form-page-header">
        <button class="btn btn-secondary btn-sm back-btn" @click="backFromPicker">
          <BrainIcon name="Chevron" :size="14" />
          Back
        </button>
        <h3>Choose a provider</h3>
      </div>
      <p class="wizard-hint">
        Pick your AI provider. We'll pre-fill the connection details and fetch its models for you.
      </p>
      <div class="provider-grid">
        <button
          v-for="t in catalogTiles"
          :key="t.id"
          type="button"
          class="provider-tile"
          @click="selectTile(t)"
        >
          <span class="provider-tile-avatar">{{ avatar(t.name) }}</span>
          <span class="provider-tile-name">{{ t.name }}</span>
          <span class="provider-tile-platform">{{
            t.platform === 'openai_compatible' ? 'OpenAI-compatible' : t.platform
          }}</span>
        </button>
      </div>
    </div>
  </template>

  <!-- ── Form ──────────────────────────────────────────────────── -->
  <template v-else-if="mode === 'form'">
    <div class="provider-wizard">
      <div class="form-page-header">
        <button class="btn btn-secondary btn-sm back-btn" @click="backFromForm">
          <BrainIcon name="Chevron" :size="14" />
          {{ isEditing ? 'Back' : 'Providers' }}
        </button>
        <h3>{{ isEditing ? 'Edit Provider' : `Set up ${preset?.name ?? ''}` }}</h3>
      </div>

      <form class="wizard-form" autocomplete="off" @submit.prevent="saveProvider">
        <!-- Name -->
        <div class="form-group">
          <label for="pName">Name</label>
          <input
            id="pName"
            v-model="formName"
            type="text"
            required
          />
        </div>

        <!-- Host / Base URL -->
        <div
          id="hostGroup"
          class="form-group wizard-step"
          :hidden="!showHostGroup"
        >
          <label for="pHost">Host / Base URL</label>
          <input
            id="pHost"
            v-model="formHost"
            type="text"
            placeholder="https://…"
          />
        </div>

        <!-- API Key -->
        <div
          id="keyGroup"
          class="form-group wizard-step"
          :hidden="!showKeyGroup"
        >
          <label for="pKey">API Key</label>
          <input
            id="pKey"
            v-model="formKey"
            type="password"
            autocomplete="new-password"
            :placeholder="isEditing ? 'Leave blank to keep existing' : 'Paste your API key'"
          />
        </div>

        <!-- Model -->
        <div
          id="modelGroup"
          class="form-group wizard-step"
          :hidden="!showModelGroup"
        >
          <label for="pModel">Model</label>
          <select id="pModel" v-model="formModel">
            <option value="">Select model…</option>
            <option v-for="m in models" :key="m" :value="m">{{ m }}</option>
          </select>
          <span :class="modelStatusClass">{{ modelStatus }}</span>
        </div>

        <div class="form-actions">
          <button type="button" class="btn btn-secondary" @click="cancelForm">Cancel</button>
          <button
            type="button"
            class="btn btn-secondary"
            :disabled="testing"
            @click="testConnection"
          >
            {{ testing ? 'Testing…' : 'Test' }}
          </button>
          <button type="submit" class="btn btn-primary" :disabled="saveDisabled">Save</button>
        </div>
      </form>
    </div>
  </template>
</template>
