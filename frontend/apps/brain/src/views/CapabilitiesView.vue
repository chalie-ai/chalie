<script setup lang="ts">
import { computed, onUnmounted, reactive, ref } from 'vue';
import type { Capability, CapabilityField, CapabilityStatus, SetupFailureReason } from '../api/capabilities';
import { capabilities } from '../api/capabilities';
import { apiErrorMessage } from '../api/http';
import { HttpError } from '@chalie/shared';
import { useToast } from '../composables/useToast';
import { useBrainResource } from '../composables/useBrainResource';
import ToggleSwitch from '../ui/ToggleSwitch.vue';
import { ChevronLeft, ChevronRight, Settings } from '@lucide/vue';

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
// Why the last setup attempt failed, shown on the form.
const formError = ref<string | null>(null);
// While a setup is still being confirmed in the background: what it waits for.
const waitingNote = ref<string | null>(null);
// The code that setup waits for the user to send, shown large on its own.
const waitingCode = ref<string | null>(null);

// How often the form asks whether a setup still being confirmed has ended.
const STATUS_POLL_MS = 3000;
let statusTimer: ReturnType<typeof setTimeout> | null = null;

function stopWaiting(): void {
  if (statusTimer) clearTimeout(statusTimer);
  statusTimer = null;
  waitingNote.value = null;
  waitingCode.value = null;
}

onUnmounted(stopWaiting);

function isConnected(c: Capability): boolean {
  return !!(c.connected || c.status === 'connected');
}

// A capability that cannot work where Chalie runs shows its reason in place of setup.
function isAvailable(c: Capability): boolean {
  return c.available !== false;
}

const formFields = computed<CapabilityField[]>(() => formCap.value?.fields ?? []);

function isVisible(f: CapabilityField): boolean {
  return (
    !f.show_when ||
    Object.entries(f.show_when).every(([k, v]) => (Array.isArray(v) ? v.includes(textValues[k]) : textValues[k] === v))
  );
}

async function openForm(c: Capability): Promise<void> {
  stopWaiting();
  formError.value = null;
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

function closeForm(): void {
  // A setup that was tried changed the card, so the list is read again.
  const tried = waitingNote.value !== null || formError.value !== null;
  stopWaiting();
  viewMode.value = 'list';
  if (tried) void load();
}

function clearSecrets(): void {
  for (const f of formFields.value) if (f.type === 'password') textValues[f.name] = '';
}

// Why setup failed, as the capability's own note says it when it has one.
async function failureNote(id: string): Promise<string | null> {
  try {
    const s = await capabilities.status(id);
    return s.connected || s.pending ? null : (s.status_note ?? null);
  } catch {
    return null;
  }
}

// What the reader is told for each reason the server gives for failed settings.
const SETUP_FAILURES: Record<SetupFailureReason, string> = {
  credentials_rejected: 'The server rejected these credentials. Check them and try again.',
  certificate_untrusted: "The server's certificate isn't trusted. If it uses a self-signed certificate, turn off Verify SSL.",
  unreachable: "Couldn't reach the server. Check the URL and that the server is on.",
  authenticator_key_missing:
    'This UI.com account asks for a code from an authenticator app. Add its setup key and try again.',
  authenticator_key_invalid:
    "This setup key isn't one an authenticator app accepts. Copy it again from your UI.com account.",
  authenticator_code_refused:
    "The UI.com account didn't accept the code made from this setup key. Check the key, and that this computer's clock is right.",
  too_many_attempts: 'UI.com has blocked sign-ins for now after too many attempts. Wait a few minutes, then try again.',
  invalid_configuration: "These settings weren't accepted. Check each field and try again.",
};

// The setup key field shows only on the UI.com account choice, so on Username & Password the reader is told where it is.
const KEY_ON_UI_ACCOUNT = 'Choose UI.com account below to add it.';

// Why the settings failed, as the server's reason for it says, when it gives one.
function setupFailure(e: unknown): string | null {
  if (!(e instanceof HttpError)) return null;
  const reason = (e.body as { reason?: string } | null | undefined)?.reason;
  if (reason === undefined || !Object.hasOwn(SETUP_FAILURES, reason)) return null;
  const text = SETUP_FAILURES[reason as SetupFailureReason];
  return reason === 'authenticator_key_missing' && textValues.auth_method === 'credentials'
    ? `${text} ${KEY_ON_UI_ACCOUNT}`
    : text;
}

async function submit(): Promise<void> {
  const cap = formCap.value;
  if (!cap) return;
  const body: Record<string, unknown> = {};
  for (const f of formFields.value) {
    const text = textValues[f.name] ?? '';
    // A password goes exactly as typed: spaces at its ends can be part of it.
    if (f.type === 'checkbox') body[f.name] = !!boolValues[f.name];
    else body[f.name] = f.type === 'password' ? text : text.trim();
  }
  formError.value = null;
  try {
    await capabilities.setup(cap.id, body);
  } catch (e) {
    formError.value = (await failureNote(cap.id)) ?? setupFailure(e) ?? apiErrorMessage(e, 'Setup failed');
    clearSecrets();
    return;
  }
  await followSetup(cap);
}

// Read the setup's outcome; while it is still being confirmed, show what it
// waits for and ask again shortly.
async function followSetup(cap: Capability): Promise<void> {
  statusTimer = null;
  let s: CapabilityStatus;
  try {
    s = await capabilities.status(cap.id);
  } catch (e) {
    waitingNote.value = null;
    waitingCode.value = null;
    formError.value = apiErrorMessage(e, 'Could not read the setup status');
    return;
  }
  // The reader left this form while the status was being read.
  if (formCap.value !== cap || viewMode.value !== 'form') return;
  if (s.pending) {
    waitingNote.value = s.status_note ?? 'Waiting for the setup to be confirmed…';
    waitingCode.value = s.setup_code ?? null;
    statusTimer = setTimeout(() => void followSetup(cap), STATUS_POLL_MS);
    return;
  }
  waitingNote.value = null;
  waitingCode.value = null;
  if (s.connected) {
    showToast(formConnected.value ? 'Settings saved' : 'Capability connected', 'success');
    viewMode.value = 'list';
    await load();
    return;
  }
  formError.value = s.status_note ?? 'Setup failed';
  clearSecrets();
}

async function cancelSetup(c: Capability): Promise<void> {
  stopWaiting();
  try {
    await capabilities.disconnect(c.id);
  } catch (e) {
    showToast(e instanceof HttpError ? 'Failed to cancel the setup' : 'Network error', 'error');
  }
  viewMode.value = 'list';
  await load();
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
        <button class="btn btn-sm back-btn" @click="closeForm">
          <ChevronLeft :size="16" /> Back
        </button>
        <h3>{{ formCap.name }} {{ formConnected ? 'Settings' : 'Setup' }}</h3>
      </div>
      <template v-if="waitingNote">
        <p class="wizard-hint">{{ waitingNote }}</p>
        <p v-if="waitingCode" class="setup-code">{{ waitingCode }}</p>
        <p class="wizard-hint">Waiting…</p>
        <div class="form-actions">
          <button type="button" class="btn" @click="cancelSetup(formCap)">Cancel</button>
        </div>
      </template>
      <template v-else>
        <p v-if="formCap.description" class="wizard-hint">{{ formCap.description }}</p>
        <p v-if="formError" class="wizard-hint wizard-error" role="alert">{{ formError }}</p>
        <form @submit.prevent="submit">
          <template v-for="f in formFields" :key="f.name">
            <div v-if="f.type === 'checkbox'" class="form-group">
              <ToggleSwitch v-model="boolValues[f.name]" :label="f.label" />
            </div>

            <div
              v-else-if="f.type === 'select'"
              class="form-group"
              role="group"
              :aria-labelledby="'cap-group-' + f.name"
            >
              <span :id="'cap-group-' + f.name" class="form-group-label">{{ f.label }}</span>
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
              <label :for="'cap-field-' + f.name">{{ f.label }}</label>
              <textarea
                :id="'cap-field-' + f.name"
                v-model="textValues[f.name]"
                rows="4"
                :placeholder="f.placeholder ?? ''"
              ></textarea>
            </div>

            <div v-else v-show="isVisible(f)" class="form-group">
              <label :for="'cap-field-' + f.name">{{ f.label }}</label>
              <input
                :id="'cap-field-' + f.name"
                v-model="textValues[f.name]"
                :type="f.type === 'password' ? 'password' : 'text'"
                :placeholder="f.placeholder ?? ''"
                :required="!!(f.required && !formConnected)"
              />
              <details v-if="f.guide" class="field-guide">
                <summary><ChevronRight class="field-guide-chev" :size="16" />How to set this up</summary>
                <div class="field-guide-body">
                  <ol>
                    <li v-for="(s, i) in f.guide.steps" :key="i">{{ s }}</li>
                  </ol>
                  <p v-for="(n, i) in f.guide.notes" :key="i">{{ n }}</p>
                </div>
              </details>
            </div>
          </template>

          <div class="form-actions">
            <button type="button" class="btn" @click="closeForm">Cancel</button>
            <button type="submit" class="btn btn-primary">
              {{ formConnected ? 'Save' : 'Connect' }}
            </button>
          </div>
        </form>
      </template>
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
    <div
      v-for="c in caps"
      :key="c.id"
      class="cap-card"
      :class="{ connected: isConnected(c), unavailable: !isAvailable(c) }"
    >
      <div class="cap-body">
        <div class="cap-header">
          <div class="cap-name">{{ c.name }}</div>
          <span class="status-dot" :class="{ active: isConnected(c) }"></span>
        </div>
        <div class="cap-meta">{{ c.platform || '' }}</div>
        <div v-if="c.version" class="cap-version">{{ c.version }}</div>
        <div v-if="c.status_note" class="cap-meta">{{ c.status_note }}</div>
        <div v-if="c.setup_code" class="setup-code setup-code--card">{{ c.setup_code }}</div>
        <div v-if="isAvailable(c)" class="cap-actions">
          <template v-if="isConnected(c)">
            <button class="btn btn-sm" @click="openForm(c)">Edit</button>
            <button class="btn btn-sm btn-danger" @click="disconnect(c)">Disconnect</button>
          </template>
          <template v-else>
            <button class="btn btn-sm" @click="openForm(c)">Setup</button>
          </template>
        </div>
      </div>
      <p v-if="!isAvailable(c)" class="cap-unavailable">{{ c.unavailable_reason }}</p>
    </div>
  </div>
</template>
