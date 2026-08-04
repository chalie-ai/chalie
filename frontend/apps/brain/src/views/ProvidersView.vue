<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from 'vue';
import type { CatalogEntry, Provider } from '../api/providers';
import { providers } from '../api/providers';
import { system } from '../api';
import { apiErrorMessage } from '../api/http';
import { useToast } from '../composables/useToast';
import { useBrainAction } from '../composables/useBrainAction';
import { useConfirm } from '../composables/useConfirm';
import { useShellStore } from '../stores/shell';
import { ChevronLeft, LayoutGrid, Plus } from '@lucide/vue';

const { show: showToast } = useToast();
const { run } = useBrainAction();
const { confirm } = useConfirm();
const shell = useShellStore();

const providerList = ref<Provider[]>([]);
const selectedId = ref<number | null>(null);
const loading = ref(true);

// Role selectors. source: 'explicit' (user-pinned) | 'auto' (fell back to main) | 'none'.
const visionId = ref<number | null>(null);
const visionSource = ref('none');
const delegateId = ref<number | null>(null);
const delegateSource = ref('none');

const catalog = ref<CatalogEntry[]>([]);
const catalogLoaded = ref(false);

const mode = ref<'list' | 'picker' | 'form'>('list');

const preset = ref<CatalogEntry | null>(null);
const editingId = ref<number | null>(null);
const editModel = ref('');

const formName = ref('');
const formHost = ref('');
const formKey = ref('');
const formModel = ref('');
const formContextWindow = ref<number | null>(null);

const models = ref<string[]>([]);
const modelsFetchInFlight = ref(false);
const modelStatus = ref('');
const modelStatusClass = ref('');
const lastFetchKey = ref('');
let modelFetchTimer: ReturnType<typeof setTimeout> | null = null;

const MODEL_FETCH_DEBOUNCE_MS = 600;

const testing = ref(false);

const needsHost = computed(() => !!preset.value?.needs_host);

const needsKey = computed(() => !!preset.value?.needs_key);

const hostReady = computed(() => !needsHost.value || formHost.value.trim() !== '');

const keyReady = computed(() => !needsKey.value || formKey.value.trim() !== '');

const canFetch = computed(() => hostReady.value && keyReady.value);

const showKeyGroup = computed(() => needsKey.value && hostReady.value);

const showModelGroup = computed(() => canFetch.value || models.value.length > 0);

const saveDisabled = computed(() => !formModel.value);

const isEditing = computed(() => editingId.value !== null);

const catalogTiles = computed(() => catalog.value);

function avatar(name: string): string {
  const s = (name || '').trim();
  return s ? s[0].toUpperCase() : '?';
}

function platformLabel(platform: string): string {
  return catalog.value.find((c) => c.platform === platform)?.name || platform;
}

onMounted(async () => {
  await load();
});

onUnmounted(() => {
  if (modelFetchTimer !== null) {
    clearTimeout(modelFetchTimer);
    modelFetchTimer = null;
  }
});

// Never re-sets `loading` to true: reloads after CRUD must not re-flash the
// initial-mount placeholder, only the first mount shows it.
async function load(): Promise<void> {
  try {
    const [provRes, selRes, visRes, delRes] = await Promise.all([
      providers.list(),
      providers.getSelected(),
      providers.getVision(),
      providers.getDelegate(),
    ]);
    providerList.value = provRes;
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
  const { ok, data: res } = await run(() => providers.setVision(pid), {
    success: 'Vision provider updated',
    failMsg: 'Failed',
    networkMsg: 'Failed to set vision provider',
  });
  if (ok && res) {
    visionId.value = res.provider ? res.provider.id : null;
    visionSource.value = res.source || (pid == null ? 'auto' : 'explicit');
  }
}

async function setDelegate(value: string): Promise<void> {
  const pid = value === '' ? null : Number(value);
  const { ok, data: res } = await run(() => providers.setDelegate(pid), {
    success: 'Delegate provider updated',
    failMsg: 'Failed',
    networkMsg: 'Failed to set delegate provider',
  });
  if (ok && res) {
    delegateId.value = res.provider ? res.provider.id : null;
    delegateSource.value = res.source || (pid == null ? 'auto' : 'explicit');
  }
}

async function selectProvider(id: number): Promise<void> {
  const { ok } = await run(() => providers.setSelected(id), {
    success: 'Provider selected',
    failMsg: 'Failed',
    networkMsg: 'Failed to select provider',
  });
  if (ok) {
    selectedId.value = id;
  }
}

// Roles a provider currently holds (mirrors backend). A provider in any role
// cannot be deleted until the role is cleared/reassigned.
function providerRoles(id: number): string[] {
  const roles: string[] = [];
  if (id === selectedId.value) roles.push('main');
  if (visionSource.value === 'explicit' && id === visionId.value) roles.push('vision');
  if (delegateSource.value === 'explicit' && id === delegateId.value) roles.push('delegate');
  return roles;
}

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
  const { ok } = await run(() => providers.delete(id), {
    success: 'Provider deleted',
    failMsg: 'Delete failed',
  });
  if (ok) {
    await load();
  }
}

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
    formHost.value = p.host || '';
    formKey.value = '';
    formModel.value = editModel.value;
    formContextWindow.value = p.context_window;
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
    formContextWindow.value = null;
    mode.value = 'picker';
  }
}

function presetFor(provider: Provider): CatalogEntry {
  const match = catalog.value.find((c) => c.platform === provider.platform);
  if (match) return match;
  // A row whose platform the running build no longer offers: keep it editable,
  // and ask for whatever it already carries rather than guessing its rules.
  return {
    id: 'edit',
    name: provider.name,
    platform: provider.platform,
    host: provider.host || '',
    needs_key: true,
    needs_host: !!provider.host,
  };
}

async function ensureCatalog(): Promise<void> {
  if (catalogLoaded.value) return;
  try {
    const res = await providers.getCatalog();
    catalog.value = Array.isArray(res) ? res : [];
    catalogLoaded.value = true; // only on success, so a failed load retries next open
  } catch {
    catalog.value = [];
    showToast('Could not load provider catalog', 'error');
  }
}

function selectTile(p: CatalogEntry): void {
  preset.value = { ...p };
  editModel.value = '';
  models.value = [];
  lastFetchKey.value = '';
  modelStatus.value = '';
  modelStatusClass.value = '';
  formName.value = p.name;
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
  mode.value = isEditing.value ? 'list' : 'picker';
}

function cancelForm(): void {
  mode.value = 'list';
}

watch([() => formHost.value, () => formKey.value], () => {
  if (mode.value !== 'form') return;
  // Any change to host/key refetches; redundant fetches are deduped downstream by
  // modelsFetchInFlight + lastFetchKey, so no suppression flag is needed here.
  debouncedFetchModels();
});

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

    // Keep the editModel selectable even if the provider no longer lists it.
    if (editModel.value && !models.value.includes(editModel.value)) {
      models.value = [...models.value, editModel.value];
    }

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

async function testConnection(): Promise<void> {
  if (!preset.value) return;
  testing.value = true;
  const body: {
    platform: string;
    model: string;
    host?: string;
    api_key?: string;
    provider_id?: number;
  } = {
    platform: preset.value.platform,
    model: formModel.value,
  };
  if (needsHost.value) body.host = formHost.value.trim();
  if (needsKey.value) body.api_key = formKey.value.trim();
  if (editingId.value !== null) body.provider_id = editingId.value;

  const { ok, data } = await run(() => providers.test(body), {
    failMsg: 'Test failed',
  });
  if (ok && data) {
    if (data.success) {
      showToast(data.message || 'Connection successful', 'success');
    } else {
      showToast(data.error || 'Test failed', 'error');
    }
  }
  testing.value = false;
}

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
    context_window: number | null;
  } = {
    platform: preset.value.platform,
    name: formName.value.trim(),
    model: formModel.value,
    // An emptied number input reads back as '' — normalise it to null, the
    // "ask the client" state. The backend owns the upper clamp.
    context_window: formContextWindow.value || null,
  };
  if (needsHost.value) body.host = formHost.value.trim();
  if (needsKey.value) {
    const k = formKey.value.trim();
    if (k) body.api_key = k;
  }

  const id = editingId.value;
  const { ok } = await run(
    () => (id !== null ? providers.update(id, body) : providers.create(body)),
    {
      success: id !== null ? 'Provider updated' : 'Provider added',
      failMsg: 'Save failed',
    },
  );
  if (ok) {
    await load();
    mode.value = 'list';
    if (shell.providersOnly) {
      try {
        const sd = await system.authStatus();
        if (sd.has_providers) shell.liftProvidersOnly();
      } catch {
        // keep locked
      }
    }
  }
}
</script>

<template>
  <template v-if="mode === 'list'">
    <div class="panel-header">
      <h2>LLM Providers</h2>
      <button class="btn btn-primary" @click="openWizard(null)">
        <Plus :size="14" />
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
            <LayoutGrid :size="40" />
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
          <label class="provider-radio" :aria-label="p.name">
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
                >⚠</span
              >
            </div>
            <div class="provider-meta">
              <span :class="`badge badge-${p.platform}`">{{ p.platform }}</span>
              <span
                v-if="p.supports_vision"
                class="badge badge-success"
                title="Verified image understanding"
                >Vision</span
              >
              <span>{{ p.model || 'no model' }}</span>
              <span v-if="p.host">· {{ p.host }}</span>
            </div>
          </div>
          <div class="provider-actions">
            <button class="btn btn-sm btn-secondary" @click="openWizard(p.id)">Edit</button>
            <button
              class="btn btn-sm btn-danger"
              :disabled="providerRoles(p.id).length > 0"
              :title="
                providerRoles(p.id).length > 0
                  ? `In use as the ${providerRoles(p.id).join(', ')} provider — clear or reassign this role before deleting`
                  : ''
              "
              :aria-disabled="providerRoles(p.id).length > 0"
              @click="confirmDelete(p.id)"
            >
              Delete
            </button>
          </div>
        </div>
      </template>
    </div>

    <template v-if="!loading && providerList.length > 0">
      <h4 class="section-head">Vision</h4>
      <p class="panel-desc">
        Choose which provider Chalie uses to understand images. Only providers that passed the
        vision probe can be selected.
      </p>
      <div class="form-group">
        <select
          v-if="visionCapable.length > 0"
          aria-label="Vision"
          :value="visionSelectValue"
          @change="setVision(($event.target as HTMLSelectElement).value)"
        >
          <option v-if="mainSupportsVision" value="">Use main provider</option>
          <option v-for="p in visionCapable" :key="p.id" :value="p.id">{{ p.name }}</option>
        </select>
        <select v-else aria-label="Vision" disabled>
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
          aria-label="Delegate"
          :value="delegateSelectValue"
          @change="setDelegate(($event.target as HTMLSelectElement).value)"
        >
          <option value="">Use main provider</option>
          <option v-for="p in providerList" :key="p.id" :value="p.id">{{ p.name }}</option>
        </select>
      </div>
    </template>
  </template>

  <template v-else-if="mode === 'picker'">
    <div class="provider-wizard">
      <div class="form-page-header">
        <button class="btn btn-secondary btn-sm back-btn" @click="backFromPicker">
          <ChevronLeft :size="14" />
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
          <span class="provider-tile-platform">{{ platformLabel(t.platform) }}</span>
        </button>
      </div>
    </div>
  </template>

  <template v-else-if="mode === 'form'">
    <div class="provider-wizard">
      <div class="form-page-header">
        <button class="btn btn-secondary btn-sm back-btn" @click="backFromForm">
          <ChevronLeft :size="14" />
          {{ isEditing ? 'Back' : 'Providers' }}
        </button>
        <h3>{{ isEditing ? 'Edit Provider' : `Set up ${preset?.name ?? ''}` }}</h3>
      </div>

      <form class="wizard-form" autocomplete="off" @submit.prevent="saveProvider">
        <div class="form-group">
          <label for="pName">Name</label>
          <input id="pName" v-model="formName" type="text" required />
        </div>

        <div id="hostGroup" class="form-group wizard-step" :hidden="!needsHost">
          <label for="pHost">Host / Base URL</label>
          <input id="pHost" v-model="formHost" type="text" placeholder="https://…" />
        </div>

        <div id="keyGroup" class="form-group wizard-step" :hidden="!showKeyGroup">
          <label for="pKey">API Key</label>
          <input
            id="pKey"
            v-model="formKey"
            type="password"
            autocomplete="new-password"
            :placeholder="isEditing ? 'Leave blank to keep existing' : 'Paste your API key'"
          />
        </div>

        <div id="modelGroup" class="form-group wizard-step" :hidden="!showModelGroup">
          <label for="pModel">Model</label>
          <select id="pModel" v-model="formModel">
            <option value="">Select model…</option>
            <option v-for="m in models" :key="m" :value="m">{{ m }}</option>
          </select>
          <span :class="modelStatusClass">{{ modelStatus }}</span>
        </div>

        <div id="contextWindowGroup" class="form-group wizard-step" :hidden="!showModelGroup">
          <label for="pContextWindow">Context window (tokens)</label>
          <input
            id="pContextWindow"
            v-model.number="formContextWindow"
            type="number"
            min="1"
            placeholder="Leave blank to detect automatically"
          />
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
