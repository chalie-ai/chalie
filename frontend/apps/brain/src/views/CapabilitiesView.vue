<script setup lang="ts">
import { computed, reactive, ref } from 'vue';
import type { Capability, CapabilityField } from '../api/capabilities';
import { capabilities } from '../api/capabilities';
import { apiErrorMessage } from '../api/http';
import { HttpError } from '@chalie/shared';
import { useToast } from '../composables/useToast';
import { useBrainResource } from '../composables/useBrainResource';
import { ChevronLeft, Settings } from '@lucide/vue';

const { show: showToast } = useToast();

const {
  data: caps,
  loading,
  reload: load,
} = useBrainResource(async () => await capabilities.list(), {
  initial: [] as Capability[],
  failMsg: 'Failed to load capabilities',
});

const viewMode = ref<'list' | 'form'>('list');
const formCap = ref<Capability | null>(null);
const formConnected = ref(false);
const textValues = reactive<Record<string, string>>({});
const boolValues = reactive<Record<string, boolean>>({});

function isConnected(c: Capability): boolean {
  return !!(c.connected || c.status === 'connected');
}

const formFields = computed<CapabilityField[]>(() => formCap.value?.fields ?? []);

function isVisible(f: CapabilityField): boolean {
  return !f.show_when || Object.entries(f.show_when).every(([k, v]) => textValues[k] === v);
}

async function openForm(c: Capability): Promise<void> {
  formCap.value = c;
  formConnected.value = isConnected(c);

  let config: Record<string, unknown> = {};
  if (formConnected.value) {
    try {
      const d = await capabilities.get(c.id);
      config = (d.config ?? d.settings ?? d ?? {}) as Record<string, unknown>;
    } catch {
      // swallow — empty config
    }
  }

  // Stale-guard: a newer openForm() may have superseded this one during the await above.
  if (formCap.value !== c) return;

  for (const key of Object.keys(textValues)) delete textValues[key];
  for (const key of Object.keys(boolValues)) delete boolValues[key];

  for (const f of c.fields ?? []) {
    if (f.type === 'checkbox') {
      boolValues[f.name] = config[f.name] !== undefined ? !!config[f.name] : !!f.default;
    } else if (f.type === 'password') {
      textValues[f.name] = '';
    } else {
      textValues[f.name] = String(config[f.name] ?? f.default ?? '');
    }
  }

  viewMode.value = 'form';
}

async function submit(): Promise<void> {
  if (!formCap.value) return;
  const body: Record<string, unknown> = {};
  for (const f of formFields.value) {
    body[f.name] = f.type === 'checkbox' ? !!boolValues[f.name] : (textValues[f.name] ?? '').trim();
  }
  try {
    await capabilities.setup(formCap.value.id, body);
    showToast(formConnected.value ? 'Settings saved' : 'Capability connected', 'success');
    viewMode.value = 'list';
    await load();
  } catch (e) {
    showToast(apiErrorMessage(e, 'Setup failed'), 'error');
  }
}

async function disconnect(c: Capability): Promise<void> {
  try {
    await capabilities.disconnect(c.id);
    showToast('Capability disconnected', 'success');
    await load();
  } catch (e) {
    showToast(e instanceof HttpError ? 'Failed to disconnect' : 'Network error', 'error');
  }
}
</script>

<template>
  <div class="panel-header">
    <h2>Capabilities</h2>
  </div>

  <div v-if="loading" class="loading">Loading…</div>

  <template v-else-if="viewMode === 'form' && formCap">
    <div class="provider-form-page">
      <div class="form-page-header">
        <button class="btn btn-secondary btn-sm back-btn" @click="viewMode = 'list'">
          <ChevronLeft :size="14" /> Back
        </button>
        <h3>{{ formCap.name }} {{ formConnected ? 'Settings' : 'Setup' }}</h3>
      </div>
      <form @submit.prevent="submit">
        <template v-for="f in formFields" :key="f.name">
          <div v-if="f.type === 'checkbox'" class="form-group">
            <label class="switch-label">
              <label class="switch">
                <input v-model="boolValues[f.name]" type="checkbox" />
                <span class="switch-track"></span>
              </label>
              <span>{{ f.label }}</span>
            </label>
          </div>

          <div v-else-if="f.type === 'select'" class="form-group">
            <label>{{ f.label }}</label>
            <div class="tab-select">
              <button
                v-for="o in f.options"
                :key="o.value"
                type="button"
                class="tab-btn"
                :class="{ active: textValues[f.name] === o.value }"
                @click="textValues[f.name] = o.value"
              >
                {{ o.label }}
              </button>
            </div>
          </div>

          <div v-else-if="f.type === 'textarea'" class="form-group">
            <label>{{ f.label }}</label>
            <textarea
              v-model="textValues[f.name]"
              rows="4"
              :placeholder="f.placeholder ?? ''"
            ></textarea>
          </div>

          <div v-else v-show="isVisible(f)" class="form-group">
            <label>{{ f.label }}</label>
            <input
              v-model="textValues[f.name]"
              :type="f.type === 'password' ? 'password' : 'text'"
              :placeholder="f.placeholder ?? ''"
              :required="!!(f.required && !formConnected)"
            />
          </div>
        </template>

        <div class="form-actions">
          <button type="button" class="btn btn-secondary" @click="viewMode = 'list'">Cancel</button>
          <button type="submit" class="btn btn-primary">
            {{ formConnected ? 'Save' : 'Connect' }}
          </button>
        </div>
      </form>
    </div>
  </template>

  <div v-else-if="caps.length === 0" class="empty-state">
    <div class="empty-icon">
      <Settings :size="40" />
    </div>
    <h3>No capabilities</h3>
    <p>No services are configured yet.</p>
  </div>

  <div v-else class="caps-grid">
    <div v-for="c in caps" :key="c.id" class="cap-card" :class="{ connected: isConnected(c) }">
      <div class="cap-header">
        <div class="cap-name">{{ c.name }}</div>
        <span class="status-dot" :class="{ active: isConnected(c) }"></span>
      </div>
      <div class="cap-meta">{{ c.platform || '' }}</div>
      <div v-if="c.version" class="cap-version">{{ c.version }}</div>
      <div class="cap-actions">
        <template v-if="isConnected(c)">
          <button class="btn btn-sm btn-secondary" @click="openForm(c)">Edit</button>
          <button class="btn btn-sm btn-danger" @click="disconnect(c)">Disconnect</button>
        </template>
        <template v-else>
          <button class="btn btn-sm btn-primary" @click="openForm(c)">Setup</button>
        </template>
      </div>
    </div>
  </div>
</template>
