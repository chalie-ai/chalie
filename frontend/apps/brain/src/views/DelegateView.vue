<script setup lang="ts">
import { ref, onMounted } from 'vue';
import { providers } from '../api/providers';
import type { Provider } from '../api/providers';
import { apiErrorMessage } from '../api/http';
import { useToast } from '../composables/useToast';
import BrainIcon from '../ui/BrainIcon.vue';

const { show: showToast } = useToast();

const providerList = ref<Provider[]>([]);
// Explicitly-pinned delegate id, or null when falling back to the main provider.
const delegateId = ref<number | null>(null);
const source = ref<string>('none');
const loading = ref(true);

onMounted(async () => {
  try {
    const [provRes, delRes] = await Promise.all([
      providers.list(),
      providers.getDelegate(),
    ]);
    providerList.value = provRes.providers ?? [];
    delegateId.value = delRes.provider ? delRes.provider.id : null;
    source.value = delRes.source || 'none';
  } catch {
    showToast('Failed to connect to backend', 'error');
  }
  loading.value = false;
});

// id null = clear the pin → subagent work runs on the main provider. Mirrors the
// legacy "Use main provider" option (TKT-960). The backend returns the resolved
// status, so reflect its source (it reports 'auto' when delegating to main).
async function setDelegate(id: number | null): Promise<void> {
  try {
    const res = await providers.setDelegate(id);
    delegateId.value = res.provider ? res.provider.id : null;
    source.value = res.source || (id == null ? 'auto' : 'explicit');
    showToast('Delegate provider updated', 'success');
  } catch (e: unknown) {
    showToast(apiErrorMessage(e, 'Failed', 'Failed to set delegate provider'), 'error');
  }
}

// "Use main provider" is the active choice whenever no provider is explicitly
// pinned (source is 'none' or the auto fallback to main).
function usingMain(): boolean {
  return source.value !== 'explicit';
}
</script>

<template>
  <div class="panel-header">
    <h2>Delegate Provider</h2>
  </div>
  <p class="panel-desc">
    Subagent work (web_search, web_browse, …) runs on this provider. Defaults to your main
    provider.
  </p>

  <template v-if="loading">
    <div class="loading">Loading…</div>
  </template>

  <template v-else-if="providerList.length === 0">
    <div class="empty-state">
      <div class="empty-icon">
        <BrainIcon name="Globe" :size="40" />
      </div>
      <h3>No providers</h3>
      <p>Add an LLM provider first.</p>
    </div>
  </template>

  <template v-else>
    <div class="providers-list">
      <!-- "Use main provider" — clears the explicit pin (provider_id null). -->
      <div :class="['provider-card', { selected: usingMain() }]">
        <label class="provider-radio">
          <input
            type="radio"
            name="del_prov"
            value="main"
            :checked="usingMain()"
            @change="setDelegate(null)"
          />
          <span class="radio-dot"></span>
        </label>
        <div class="provider-info">
          <div class="provider-name">Use main provider</div>
          <div class="provider-meta">
            <span class="badge badge-muted">Default</span>
            <span>Subagents share your main chat provider</span>
          </div>
        </div>
      </div>

      <div
        v-for="p in providerList"
        :key="p.id"
        :class="['provider-card', { selected: source === 'explicit' && p.id === delegateId }]"
      >
        <label class="provider-radio">
          <input
            type="radio"
            name="del_prov"
            :value="p.id"
            :checked="source === 'explicit' && p.id === delegateId"
            @change="setDelegate(p.id)"
          />
          <span class="radio-dot"></span>
        </label>
        <div class="provider-info">
          <div class="provider-name">{{ p.name }}</div>
          <div class="provider-meta">
            <span :class="`badge badge-${p.platform}`">{{ p.platform }}</span>
            <span>{{ p.model || 'no model' }}</span>
          </div>
        </div>
      </div>
    </div>
  </template>
</template>
