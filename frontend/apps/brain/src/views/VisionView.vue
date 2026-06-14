<script setup lang="ts">
import { ref, onMounted } from 'vue';
import { providers } from '../api/providers';
import type { Provider } from '../api/providers';
import { apiErrorMessage } from '../api/http';
import { useToast } from '../composables/useToast';
import BrainIcon from '../ui/BrainIcon.vue';

const { show: showToast } = useToast();

const providerList = ref<Provider[]>([]);
const visionId = ref<number | null>(null);
const source = ref<string>('none');
const loading = ref(true);

onMounted(async () => {
  try {
    const [provRes, visRes] = await Promise.all([
      providers.list(),
      providers.getVision(),
    ]);
    providerList.value = provRes.providers ?? [];
    visionId.value = visRes.provider ? visRes.provider.id : null;
    source.value = visRes.source || 'none';
  } catch {
    showToast('Failed to connect to backend', 'error');
  }
  loading.value = false;
});

async function setVision(id: number): Promise<void> {
  try {
    await providers.setVision(id);
    visionId.value = id;
    source.value = 'explicit';
    showToast('Vision provider set', 'success');
  } catch (e: unknown) {
    showToast(apiErrorMessage(e, 'Failed', 'Failed to set vision provider'), 'error');
  }
}
</script>

<template>
  <div class="panel-header">
    <h2>Vision Provider</h2>
  </div>
  <p class="panel-desc">
    Pick the provider Chalie uses to understand images. Only providers that passed the vision probe
    can be selected.
  </p>

  <template v-if="loading">
    <div class="loading">Loading…</div>
  </template>

  <template v-else-if="providerList.length === 0">
    <div class="empty-state">
      <div class="empty-icon">
        <BrainIcon name="Eye" :size="40" />
      </div>
      <h3>No providers</h3>
      <p>Add an LLM provider first.</p>
    </div>
  </template>

  <template v-else>
    <p v-if="source === 'auto' && visionId !== null" class="panel-desc">
      Auto-selected the active provider because it supports vision. Pick one explicitly to lock it
      in.
    </p>

    <div class="providers-list">
      <div
        v-for="p in providerList"
        :key="p.id"
        :class="[
          'provider-card',
          { selected: p.supports_vision && p.id === visionId },
          { disabled: !p.supports_vision },
        ]"
      >
        <label class="provider-radio">
          <input
            type="radio"
            name="vis_prov"
            :value="p.id"
            :disabled="!p.supports_vision"
            :checked="p.supports_vision && p.id === visionId"
            @change="setVision(p.id)"
          />
          <span class="radio-dot"></span>
        </label>
        <div class="provider-info">
          <div class="provider-name">{{ p.name }}</div>
          <div class="provider-meta">
            <span :class="`badge badge-${p.platform}`">{{ p.platform }}</span>
            <span>{{ p.model || 'no model' }}</span>
            <span
              v-if="p.supports_vision"
              class="badge badge-success"
            >Vision</span>
            <span
              v-else
              class="badge badge-muted"
              title="Did not pass the vision probe"
            >No vision</span>
          </div>
        </div>
      </div>
    </div>
  </template>
</template>
