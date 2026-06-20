<script setup lang="ts">
import { ref, onMounted } from 'vue';
import { mcp } from '../api/mcp';
import type { McpClient, McpServerConfig } from '../api/mcp';
import { apiErrorMessage } from '../api/http';
import { HttpError } from '@chalie/shared';
import { useToast } from '../composables/useToast';
import { useConfirm } from '../composables/useConfirm';
import { Copy } from '@lucide/vue';

const { show: showToast } = useToast();
const { confirm } = useConfirm();

// ── Inbound state ─────────────────────────────────────────────────────────
const inConfig = ref<McpServerConfig>({});
const inPort = ref<number>(8462);
const loadingInbound = ref(true);

// ── Outbound state ────────────────────────────────────────────────────────
const outServers = ref<McpClient[]>([]);
const editingId = ref<string | number | null>(null);
const loadingOutbound = ref(true);

// Add-form state
const addEnabled = ref(true);
const addName = ref('');
const addHost = ref('');
const addHeaders = ref<{ key: string; value: string }[]>([]);

// Edit-form state
const editEnabled = ref(true);
const editName = ref('');
const editHost = ref('');
const editHeaders = ref<{ key: string; value: string }[]>([]);

// ── Inbound ───────────────────────────────────────────────────────────────

async function loadInbound(): Promise<void> {
  try {
    const data = await mcp.getServerConfig();
    inConfig.value = data;
    inPort.value = (data.port as number) || 8462;
  } catch {
    showToast('Failed to load MCP inbound settings', 'error');
  } finally {
    loadingInbound.value = false;
  }
}

async function saveInbound(updates: Partial<McpServerConfig>): Promise<void> {
  try {
    await mcp.updateServerConfig(updates);
    Object.assign(inConfig.value, updates);
    showToast('Settings saved', 'success');
  } catch (e) {
    showToast(e instanceof HttpError ? 'Save failed' : 'Network error', 'error');
  }
}

function onEnabledChange(e: Event): void {
  const checked = (e.target as HTMLInputElement).checked;
  saveInbound({ enabled: checked });
}

function onPortBlur(): void {
  const val = inPort.value;
  if (val >= 1024 && val <= 65535 && val !== inConfig.value.port) {
    saveInbound({ port: val });
  }
}

async function copyToken(): Promise<void> {
  try {
    await navigator.clipboard.writeText((inConfig.value.token as string) || '');
    showToast('Token copied', 'success');
  } catch {
    showToast('Copy failed', 'error');
  }
}

async function regenToken(): Promise<void> {
  try {
    const data = await mcp.regenerateToken();
    inConfig.value = { ...inConfig.value, token: data.token };
    showToast('Token regenerated', 'success');
  } catch (e) {
    showToast(e instanceof HttpError ? 'Regenerate failed' : 'Network error', 'error');
  }
}

// ── Outbound ──────────────────────────────────────────────────────────────

async function loadOutbound(): Promise<void> {
  try {
    outServers.value = await mcp.listClients();
  } catch {
    showToast('Failed to load MCP outbound servers', 'error');
    outServers.value = [];
  } finally {
    loadingOutbound.value = false;
  }
}

function addHeaderRow(list: { key: string; value: string }[]): void {
  list.push({ key: '', value: '' });
}

function removeHeaderRow(list: { key: string; value: string }[], idx: number): void {
  list.splice(idx, 1);
}

function collectHeaders(list: { key: string; value: string }[]): Record<string, string> {
  const headers: Record<string, string> = {};
  for (const row of list) {
    const k = row.key.trim();
    const v = row.value.trim();
    if (k) headers[k] = v;
  }
  return headers;
}

async function addServer(): Promise<void> {
  const name = addName.value.trim();
  const host = addHost.value.trim();
  const enabled = addEnabled.value;
  const headers = collectHeaders(addHeaders.value);

  if (!name) { showToast('Name is required', 'error'); return; }
  if (!host) { showToast('Host is required', 'error'); return; }

  try {
    await mcp.createClient({ name, host, headers, enabled });
    showToast(`Server "${name}" added`, 'success');
    addName.value = '';
    addHost.value = '';
    addEnabled.value = true;
    addHeaders.value = [];
    await loadOutbound();
  } catch (e) {
    showToast(apiErrorMessage(e, 'Failed to add server'), 'error');
  }
}

function startEdit(server: McpClient): void {
  editingId.value = server.id;
  editEnabled.value = server.enabled !== false;
  editName.value = server.name;
  editHost.value = server.host;
  const raw = server.headers;
  editHeaders.value = raw
    ? Object.entries(raw).map(([k, v]) => ({ key: k, value: String(v) }))
    : [];
}

function cancelEdit(): void {
  editingId.value = null;
}

async function saveEdit(id: string | number): Promise<void> {
  const name = editName.value.trim();
  const host = editHost.value.trim();
  const enabled = editEnabled.value;
  const headers = collectHeaders(editHeaders.value);

  if (!name) { showToast('Name is required', 'error'); return; }
  if (!host) { showToast('Host is required', 'error'); return; }

  try {
    await mcp.updateClient(id, { name, host, headers, enabled });
    showToast(`Server "${name}" updated`, 'success');
    editingId.value = null;
    await testServer(id, true);
  } catch (e) {
    showToast(apiErrorMessage(e, 'Failed to update server'), 'error');
  }
}

async function testServer(id: string | number, silent = false): Promise<void> {
  try {
    const data = await mcp.testClient(id);
    if (!silent) {
      const msg = data.reachable
        ? `Online — ${data.tool_count} tool(s) synced`
        : `Offline (${data.status})`;
      showToast(msg, data.reachable ? 'success' : 'error');
    }
  } catch (e) {
    if (!silent) showToast(e instanceof HttpError ? 'Test request failed' : 'Network error', 'error');
  } finally {
    await loadOutbound();
  }
}

async function toggleServer(server: McpClient): Promise<void> {
  const enabled = !server.enabled;
  try {
    await mcp.updateClient(server.id, { enabled });
    showToast(enabled ? 'Server enabled' : 'Server disabled', 'success');
    await loadOutbound();
  } catch (e) {
    showToast(e instanceof HttpError ? 'Update failed' : 'Network error', 'error');
  }
}

async function deleteServer(server: McpClient): Promise<void> {
  const ok = await confirm({
    title: 'Delete MCP Server',
    desc: 'Delete this MCP server connection? This will remove its tools and policy rows.',
    confirmLabel: 'Delete',
    confirmClass: 'btn-danger',
  });
  if (!ok) return;
  try {
    await mcp.deleteClient(server.id);
    showToast('Server deleted', 'success');
    await loadOutbound();
  } catch (e) {
    showToast(e instanceof HttpError ? 'Delete failed' : 'Network error', 'error');
  }
}

onMounted(async () => {
  await Promise.all([loadInbound(), loadOutbound()]);
});
</script>

<template>
  <div class="panel-header">
    <h2>MCP</h2>
  </div>

  <!-- ── Inbound section ─────────────────────────────────────────────── -->
  <section class="mcp-section">
    <h3 class="mcp-section-title">Inbound</h3>
    <p class="panel-desc">
      External agents (Claude Code, Codex, CI bots) connect to Chalie via MCP.
      Manage access and view the connection token below.
    </p>

    <div v-if="loadingInbound" class="loading">Loading…</div>

    <div v-else class="mcp-settings">
      <div class="mcp-row">
        <label class="mcp-label">Server Enabled</label>
        <label class="switch">
          <input
            type="checkbox"
            :checked="inConfig.enabled !== false"
            @change="onEnabledChange"
          >
          <span class="switch-track"></span>
        </label>
      </div>

      <div class="form-group">
        <label for="mcpPort">Port</label>
        <div class="input-group">
          <span class="input-prefix">TCP</span>
          <input
            id="mcpPort"
            v-model.number="inPort"
            type="number"
            min="1024"
            max="65535"
            @blur="onPortBlur"
          >
        </div>
      </div>

      <div class="mcp-token-section">
        <h4>Connection Token</h4>
        <p class="mcp-hint">Give this token to external agents so they can authenticate.</p>
        <div class="input-group">
          <input
            type="text"
            class="monospace"
            :value="(inConfig.token as string) || ''"
            readonly
          >
          <button class="input-suffix-btn" title="Copy" @click="copyToken">
            <Copy :size="14" />
          </button>
        </div>
        <div style="margin-top:10px">
          <button class="btn btn-sm btn-danger" @click="regenToken">Regenerate Token</button>
        </div>
      </div>
    </div>
  </section>

  <!-- ── Outbound section ────────────────────────────────────────────── -->
  <section class="mcp-section" style="margin-top:28px">
    <h3 class="mcp-section-title">Outbound</h3>
    <p class="panel-desc">
      Connect Chalie to remote MCP servers so their tools become available.
      Chalie uses the streamable-HTTP transport.
    </p>

    <div v-if="loadingOutbound" class="loading">Loading…</div>

    <template v-else>
    <!-- Server cards -->
    <template v-for="server in outServers" :key="server.id">
      <!-- Edit mode -->
      <div v-if="editingId === server.id" class="mcp-out-card mcp-out-card-editing">
        <h4 class="mcp-out-add-title">Edit MCP Server</h4>

        <div class="mcp-out-form-row">
          <label class="mcp-label">Enabled</label>
          <label class="switch">
            <input
              type="checkbox"
              :checked="editEnabled"
              @change="editEnabled = ($event.target as HTMLInputElement).checked"
            >
            <span class="switch-track"></span>
          </label>
        </div>

        <div class="form-group">
          <label for="mcpEditName">Name</label>
          <input
            id="mcpEditName"
            v-model="editName"
            type="text"
            class="form-input"
          >
        </div>

        <div class="form-group">
          <label for="mcpEditHost">Host (incl. port)</label>
          <input
            id="mcpEditHost"
            v-model="editHost"
            type="text"
            class="form-input"
          >
        </div>

        <div class="form-group">
          <label>Additional Headers</label>
          <div class="mcp-headers-list">
            <div
              v-for="(row, idx) in editHeaders"
              :key="idx"
              class="mcp-header-row"
            >
              <input
                v-model="row.key"
                type="text"
                placeholder="Header name"
                class="form-input mcp-header-key"
              >
              <span class="mcp-header-sep">:</span>
              <input
                v-model="row.value"
                type="text"
                placeholder="Value"
                class="form-input mcp-header-val"
              >
              <button
                class="btn btn-xs btn-danger mcp-header-remove"
                type="button"
                @click="removeHeaderRow(editHeaders, idx)"
              >✕</button>
            </div>
          </div>
          <button
            class="btn btn-xs btn-secondary"
            type="button"
            @click="addHeaderRow(editHeaders)"
          >+ Header</button>
        </div>

        <div class="mcp-out-card-actions">
          <button class="btn btn-primary" type="button" @click="saveEdit(server.id)">Save</button>
          <button class="btn btn-secondary" type="button" @click="cancelEdit">Cancel</button>
        </div>
      </div>

      <!-- Display mode -->
      <div v-else class="mcp-out-card">
        <div class="mcp-out-card-header">
          <div>
            <span class="mcp-out-name">{{ server.name }}</span>
            <span class="mcp-out-host"> — {{ server.host }}</span>
          </div>
          <div class="mcp-out-badges">
            <span :class="`badge badge-status badge-${server.status}`">{{ server.status }}</span>
            <span :class="server.enabled ? 'badge badge-enabled' : 'badge badge-disabled'">
              {{ server.enabled ? 'enabled' : 'disabled' }}
            </span>
          </div>
        </div>
        <div class="mcp-out-card-actions">
          <button class="btn btn-xs btn-secondary" type="button" @click="startEdit(server)">Edit</button>
          <button class="btn btn-xs btn-secondary" type="button" @click="testServer(server.id)">Test</button>
          <button
            class="btn btn-xs"
            :class="server.enabled ? 'btn-secondary' : 'btn-primary'"
            type="button"
            @click="toggleServer(server)"
          >{{ server.enabled ? 'Disable' : 'Enable' }}</button>
          <button class="btn btn-xs btn-danger" type="button" @click="deleteServer(server)">Delete</button>
        </div>
      </div>
    </template>

    <!-- Add-server form -->
    <div class="mcp-out-add-form">
      <h4 class="mcp-out-add-title">Add Remote MCP Server</h4>

      <div class="mcp-out-form-row">
        <label class="mcp-label">Enabled</label>
        <label class="switch">
          <input
            type="checkbox"
            :checked="addEnabled"
            @change="addEnabled = ($event.target as HTMLInputElement).checked"
          >
          <span class="switch-track"></span>
        </label>
      </div>

      <div class="form-group">
        <label for="mcpOutName">Name</label>
        <input
          id="mcpOutName"
          v-model="addName"
          type="text"
          placeholder="e.g. taskie"
          class="form-input"
        >
      </div>

      <div class="form-group">
        <label for="mcpOutHost">Host (incl. port)</label>
        <input
          id="mcpOutHost"
          v-model="addHost"
          type="text"
          placeholder="https://mcp.example.com/mcp"
          class="form-input"
        >
      </div>

      <div class="form-group">
        <label>Additional Headers</label>
        <div class="mcp-headers-list">
          <div
            v-for="(row, idx) in addHeaders"
            :key="idx"
            class="mcp-header-row"
          >
            <input
              v-model="row.key"
              type="text"
              placeholder="Header name"
              class="form-input mcp-header-key"
            >
            <span class="mcp-header-sep">:</span>
            <input
              v-model="row.value"
              type="text"
              placeholder="Value"
              class="form-input mcp-header-val"
            >
            <button
              class="btn btn-xs btn-danger mcp-header-remove"
              type="button"
              @click="removeHeaderRow(addHeaders, idx)"
            >✕</button>
          </div>
        </div>
        <button
          class="btn btn-xs btn-secondary"
          type="button"
          @click="addHeaderRow(addHeaders)"
        >+ Header</button>
      </div>

      <button class="btn btn-primary" type="button" @click="addServer">Add Server</button>
    </div>
    </template>
  </section>
</template>
