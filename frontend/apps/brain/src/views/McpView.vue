<script setup lang="ts">
import { onMounted, ref } from 'vue';
import type { McpClient, McpServerConfig } from '../api/mcp';
import { mcp } from '../api/mcp';
import { useBrainAction } from '../composables/useBrainAction';
import { HttpError } from '@chalie/shared';
import { useToast } from '../composables/useToast';
import { useConfirm } from '../composables/useConfirm';
import { type HeaderRow, addHeaderRow, removeHeaderRow, collectHeaders } from './mcpHeaders';

const { show: showToast } = useToast();
const { run } = useBrainAction();
const { confirm } = useConfirm();

const inConfig = ref<McpServerConfig>({});
const inPort = ref<number>(8462);
const loadingInbound = ref(true);

const outServers = ref<McpClient[]>([]);
const editingId = ref<string | number | null>(null);
const loadingOutbound = ref(true);

const addEnabled = ref(true);
const addName = ref('');
const addHost = ref('');
const addHeaders = ref<HeaderRow[]>([]);

const editEnabled = ref(true);
const editName = ref('');
const editHost = ref('');
const editHeaders = ref<HeaderRow[]>([]);

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
  const { ok } = await run(() => mcp.updateServerConfig(updates), {
    success: 'Settings saved',
    failMsg: 'Save failed',
  });
  // The update is applied to the listener before the 204 returns, so a fresh
  // read shows whether it is actually listening on the new settings.
  if (ok) await loadInbound();
}

function onEnabledChange(e: Event): void {
  saveInbound({ enabled: (e.target as HTMLInputElement).checked });
}

function onPortBlur(): void {
  const val = inPort.value;
  if (val >= 1024 && val <= 65535 && val !== inConfig.value.port) {
    saveInbound({ port: val });
  }
}

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

async function addServer(): Promise<void> {
  const name = addName.value.trim();
  const host = addHost.value.trim();
  const headers = collectHeaders(addHeaders.value);

  if (!name) {
    showToast('Name is required', 'error');
    return;
  }
  if (!host) {
    showToast('Host is required', 'error');
    return;
  }

  const { ok } = await run(
    () => mcp.createClient({ name, host, headers, enabled: addEnabled.value }),
    { success: `Server "${name}" added`, failMsg: 'Failed to add server' },
  );
  if (ok) {
    addName.value = '';
    addHost.value = '';
    addEnabled.value = true;
    addHeaders.value = [];
    await loadOutbound();
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
  const headers = collectHeaders(editHeaders.value);

  if (!name) {
    showToast('Name is required', 'error');
    return;
  }
  if (!host) {
    showToast('Host is required', 'error');
    return;
  }

  const { ok } = await run(
    () => mcp.updateClient(id, { name, host, headers, enabled: editEnabled.value }),
    { success: `Server "${name}" updated`, failMsg: 'Failed to update server' },
  );
  if (ok) {
    editingId.value = null;
    await testServer(id, true);
  }
}

async function testServer(id: string | number, silent = false): Promise<void> {
  try {
    const data = await mcp.testClient(id);
    if (!silent) {
      const msg = data.connected
        ? `Connected — ${data.tool_count} tool(s) synced`
        : data.error ? `Offline — ${data.error}` : 'Offline';
      showToast(msg, data.connected ? 'success' : 'error');
    }
  } catch (e) {
    if (!silent)
      showToast(e instanceof HttpError ? 'Test request failed' : 'Network error', 'error');
  } finally {
    await loadOutbound();
  }
}

async function toggleServer(server: McpClient): Promise<void> {
  const enabled = !server.enabled;
  const { ok } = await run(() => mcp.updateClient(server.id, { enabled }), {
    success: enabled ? 'Server enabled' : 'Server disabled',
    failMsg: 'Update failed',
  });
  if (ok) await loadOutbound();
}

async function deleteServer(server: McpClient): Promise<void> {
  const ok = await confirm({
    title: 'Delete MCP Server',
    desc: 'Delete this MCP server connection? This will remove its tools and policy rows.',
    confirmLabel: 'Delete',
    confirmClass: 'btn-danger',
  });
  if (!ok) return;
  const { ok: deleted } = await run(() => mcp.deleteClient(server.id), {
    success: 'Server deleted',
    failMsg: 'Delete failed',
  });
  if (deleted) await loadOutbound();
}

onMounted(async () => {
  await Promise.all([loadInbound(), loadOutbound()]);
});
</script>

<template>
  <div class="panel-header">
    <h2>MCP</h2>
  </div>

  <section class="mcp-section">
    <h3 class="mcp-section-title">Inbound</h3>
    <p class="panel-desc">
      External agents (Claude Code, Codex, CI bots) connect to Chalie via MCP. What each one may
      do is set per tool under Policies › External agent.
    </p>

    <div v-if="loadingInbound" class="loading">Loading…</div>

    <div v-else class="mcp-settings">
      <div class="mcp-row">
        <label class="mcp-label" for="mcpInboundEnabled">Server Enabled</label>
        <label class="switch" aria-label="Server Enabled">
          <input
            id="mcpInboundEnabled"
            type="checkbox"
            :checked="inConfig.enabled !== false"
            @change="onEnabledChange"
          />
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
          />
        </div>
      </div>

      <p class="mcp-hint" :class="{ 'mcp-status-error': inConfig.error }" aria-live="polite">
        <template v-if="inConfig.listening">Listening on port {{ inConfig.listening_port }}</template>
        <template v-else-if="inConfig.error">Not listening — {{ inConfig.error }}</template>
        <template v-else>Not listening</template>
      </p>
    </div>
  </section>

  <section class="mcp-section mt-lg">
    <h3 class="mcp-section-title">Outbound</h3>
    <p class="panel-desc">
      Connect Chalie to remote MCP servers so their tools become available. Chalie uses the
      streamable-HTTP transport.
    </p>

    <div v-if="loadingOutbound" class="loading">Loading…</div>

    <template v-else>
      <template v-for="server in outServers" :key="server.id">
        <div v-if="editingId === server.id" class="mcp-out-card mcp-out-card-editing">
          <h4 class="mcp-out-add-title">Edit MCP Server</h4>

          <div class="mcp-out-form-row">
            <label class="mcp-label" for="mcpEditEnabled">Enabled</label>
            <label class="switch" aria-label="Enabled">
              <input
                id="mcpEditEnabled"
                type="checkbox"
                :checked="editEnabled"
                @change="editEnabled = ($event.target as HTMLInputElement).checked"
              />
              <span class="switch-track"></span>
            </label>
          </div>

          <div class="form-group">
            <label for="mcpEditName">Name</label>
            <input id="mcpEditName" v-model="editName" type="text" class="form-input" />
          </div>

          <div class="form-group">
            <label for="mcpEditHost">Host (incl. port)</label>
            <input id="mcpEditHost" v-model="editHost" type="text" class="form-input" />
          </div>

          <div class="form-group" role="group" aria-labelledby="mcpEditHeadersLabel">
            <span id="mcpEditHeadersLabel" class="form-group-label">Additional Headers</span>
            <div class="mcp-headers-list">
              <div v-for="(row, idx) in editHeaders" :key="idx" class="mcp-header-row">
                <input
                  v-model="row.key"
                  type="text"
                  placeholder="Header name"
                  aria-label="Header name"
                  class="form-input mcp-header-key"
                />
                <span class="mcp-header-sep">:</span>
                <input
                  v-model="row.value"
                  type="text"
                  placeholder="Value"
                  aria-label="Value"
                  class="form-input mcp-header-val"
                />
                <button
                  class="btn btn-xs btn-danger mcp-header-remove"
                  type="button"
                  @click="removeHeaderRow(editHeaders, idx)"
                >
                  ✕
                </button>
              </div>
            </div>
            <button
              class="btn btn-xs btn-secondary"
              type="button"
              @click="addHeaderRow(editHeaders)"
            >
              + Header
            </button>
          </div>

          <div class="mcp-out-card-actions">
            <button class="btn btn-primary" type="button" @click="saveEdit(server.id)">Save</button>
            <button class="btn btn-secondary" type="button" @click="cancelEdit">Cancel</button>
          </div>
        </div>

        <div v-else class="mcp-out-card">
          <div class="mcp-out-card-header">
            <div>
              <span class="mcp-out-name">{{ server.name }}</span>
              <span class="mcp-out-host"> — {{ server.host }}</span>
            </div>
            <div class="mcp-out-badges">
              <span :class="server.connected ? 'badge badge-status badge-online' : 'badge badge-status badge-offline'">{{ server.connected ? 'connected' : 'offline' }}</span>
              <span :class="server.enabled ? 'badge badge-enabled' : 'badge badge-disabled'">
                {{ server.enabled ? 'enabled' : 'disabled' }}
              </span>
            </div>
          </div>
          <div class="mcp-out-card-actions">
            <button class="btn btn-xs btn-secondary" type="button" @click="startEdit(server)">
              Edit
            </button>
            <button class="btn btn-xs btn-secondary" type="button" @click="testServer(server.id)">
              Test
            </button>
            <button
              class="btn btn-xs"
              :class="server.enabled ? 'btn-secondary' : 'btn-primary'"
              type="button"
              @click="toggleServer(server)"
            >
              {{ server.enabled ? 'Disable' : 'Enable' }}
            </button>
            <button class="btn btn-xs btn-danger" type="button" @click="deleteServer(server)">
              Delete
            </button>
          </div>
        </div>
      </template>

      <div class="mcp-out-add-form">
        <h4 class="mcp-out-add-title">Add Remote MCP Server</h4>

        <div class="mcp-out-form-row">
          <label class="mcp-label" for="mcpAddEnabled">Enabled</label>
          <label class="switch" aria-label="Enabled">
            <input
              id="mcpAddEnabled"
              type="checkbox"
              :checked="addEnabled"
              @change="addEnabled = ($event.target as HTMLInputElement).checked"
            />
            <span class="switch-track"></span>
          </label>
        </div>

        <div class="form-group">
          <label for="mcpOutName">Name</label>
          <input
            id="mcpOutName"
            v-model="addName"
            type="text"
            placeholder="e.g. my-server"
            class="form-input"
          />
        </div>

        <div class="form-group">
          <label for="mcpOutHost">Host (incl. port)</label>
          <input
            id="mcpOutHost"
            v-model="addHost"
            type="text"
            placeholder="https://mcp.example.com/mcp"
            class="form-input"
          />
        </div>

        <div class="form-group" role="group" aria-labelledby="mcpAddHeadersLabel">
          <span id="mcpAddHeadersLabel" class="form-group-label">Additional Headers</span>
          <div class="mcp-headers-list">
            <div v-for="(row, idx) in addHeaders" :key="idx" class="mcp-header-row">
              <input
                v-model="row.key"
                type="text"
                placeholder="Header name"
                aria-label="Header name"
                class="form-input mcp-header-key"
              />
              <span class="mcp-header-sep">:</span>
              <input
                v-model="row.value"
                type="text"
                placeholder="Value"
                aria-label="Value"
                class="form-input mcp-header-val"
              />
              <button
                class="btn btn-xs btn-danger mcp-header-remove"
                type="button"
                @click="removeHeaderRow(addHeaders, idx)"
              >
                ✕
              </button>
            </div>
          </div>
          <button class="btn btn-xs btn-secondary" type="button" @click="addHeaderRow(addHeaders)">
            + Header
          </button>
        </div>

        <button class="btn btn-primary" type="button" @click="addServer">Add Server</button>
      </div>
    </template>
  </section>
</template>
