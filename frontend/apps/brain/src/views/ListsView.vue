<script setup lang="ts">
import { computed, reactive, ref } from 'vue';
import type { List, ListItem } from '../api/lists';
import { lists as listsApi } from '../api/lists';
import { useToast } from '../composables/useToast';
import { useConfirm } from '../composables/useConfirm';
import { useBrainResource } from '../composables/useBrainResource';
import BrainModal from '../ui/BrainModal.vue';
import { ChevronDown, ChevronRight, List as ListIcon, Plus, X } from '@lucide/vue';
import { HttpError } from '@chalie/shared';

const { show: showToast } = useToast();
const { confirm } = useConfirm();

const {
  data: listsData,
  loading,
  reload: load,
} = useBrainResource(async () => await listsApi.list(), {
  initial: [] as List[],
  failMsg: 'Failed to load lists',
});

const expanded = reactive<Record<string, boolean>>({});

const showNew = ref(false);
const newListName = ref('');
const showRename = ref(false);
const renameName = ref('');
const renameId = ref<string | number | null>(null);

function counts(list: List): { done: number; total: number; pct: number } {
  const items = list.items ?? [];
  const done = list.items ? items.filter((i) => i.checked).length : (list.checked_count ?? 0);
  const total = list.items ? items.length : (list.item_count ?? 0);
  const pct = total > 0 ? Math.round((done / total) * 100) : 0;
  return { done, total, pct };
}

// Reuses the same `list` reference from listsData so toggle/fetchListDetail mutations still apply.
const decoratedLists = computed(() => listsData.value.map((list) => ({ list, c: counts(list) })));

async function fetchListDetail(list: List): Promise<void> {
  try {
    list.items = await listsApi.items(list.id);
  } catch {
    showToast('Failed to load list items', 'error');
  }
}

function toggle(list: List): void {
  const id = String(list.id);
  expanded[id] = !expanded[id];
  if (expanded[id]) fetchListDetail(list);
}

async function toggleItem(list: List, item: ListItem, checked: boolean): Promise<void> {
  try {
    await listsApi.toggleItem(list.id, item.id, checked);
    item.checked = checked;
  } catch {
    showToast('Failed to update item', 'error');
  }
}

function onAddKey(e: KeyboardEvent, list: List): void {
  const input = e.target as HTMLInputElement;
  const value = input.value.trim();
  if (value) {
    addItem(list, value);
    input.value = '';
  }
}

async function addItem(list: List, text: string): Promise<void> {
  try {
    await listsApi.addItem(list.id, text);
    await fetchListDetail(list);
  } catch (e) {
    // silent on non-ok; toast only on network error
    if (!(e instanceof HttpError)) showToast('Failed to add item', 'error');
  }
}

function openRename(list: List): void {
  renameId.value = list.id;
  renameName.value = list.name;
  showRename.value = true;
}

async function createList(): Promise<void> {
  try {
    await listsApi.create(newListName.value.trim());
    showNew.value = false;
    showToast('List created', 'success');
    await load();
  } catch (e) {
    showToast(e instanceof HttpError ? 'Failed to create list' : 'Network error', 'error');
  }
}

async function renameList(): Promise<void> {
  if (renameId.value == null) return;
  try {
    await listsApi.rename(renameId.value, renameName.value.trim());
    showRename.value = false;
    showToast('List renamed', 'success');
    await load();
  } catch (e) {
    showToast(e instanceof HttpError ? 'Failed to rename' : 'Network error', 'error');
  }
}

async function deleteList(list: List): Promise<void> {
  const ok = await confirm({
    title: 'Delete List',
    desc: 'Delete "' + list.name + '"?',
    confirmLabel: 'Delete',
    confirmClass: 'btn-danger',
  });
  if (!ok) return;
  try {
    await listsApi.delete(list.id);
    showToast('List deleted', 'success');
    await load();
  } catch (e) {
    showToast(e instanceof HttpError ? 'Delete failed' : 'Network error', 'error');
  }
}
</script>

<template>
  <div class="panel-header">
    <h2>Lists</h2>
    <button
      class="btn btn-primary"
      @click="
        showNew = true;
        newListName = '';
      "
    >
      <Plus :size="14" /> New List
    </button>
  </div>

  <div v-if="loading" class="loading">Loading…</div>

  <div v-else-if="listsData.length === 0" class="empty-state">
    <div class="empty-icon">
      <ListIcon :size="40" />
    </div>
    <h3>No lists</h3>
    <p>Create your first list to get started.</p>
  </div>

  <template v-else>
    <div v-for="{ list, c } in decoratedLists" :key="list.id" class="list-card">
      <div class="list-card-header" @click="toggle(list)">
        <div class="list-card-title">
          <span class="list-chev">
            <component :is="expanded[String(list.id)] ? ChevronDown : ChevronRight" :size="14" />
          </span>
          <span>{{ list.name }}</span>
          <span class="list-count">{{ c.done }}/{{ c.total }}</span>
        </div>
        <div class="list-card-actions">
          <button class="btn btn-sm btn-secondary" @click.stop="openRename(list)">Rename</button>
          <button class="btn btn-sm btn-danger" @click.stop="deleteList(list)">Delete</button>
        </div>
      </div>

      <div v-if="c.total > 0" class="progress-bar">
        <div class="progress-fill" :style="{ '--fill': c.pct + '%' }"></div>
      </div>

      <div v-if="expanded[String(list.id)]" class="list-items">
        <label v-for="item in list.items ?? []" :key="item.id" class="list-item">
          <input
            type="checkbox"
            :checked="item.checked"
            @change="toggleItem(list, item, ($event.target as HTMLInputElement).checked)"
          />
          <span :class="{ done: item.checked }">{{ item.content }}</span>
        </label>
        <div class="list-add-row">
          <input
            type="text"
            class="list-add-input"
            placeholder="Add item…"
            maxlength="300"
            @keydown.enter="onAddKey($event, list)"
          />
        </div>
      </div>
    </div>
  </template>

  <BrainModal v-model="showNew" size="sm">
    <div class="modal-header">
      <h3>New List</h3>
      <button class="btn-close" @click="showNew = false">
        <X :size="16" />
      </button>
    </div>
    <form @submit.prevent="createList">
      <div class="form-group">
        <label for="newListName">List Name</label>
        <input
          id="newListName"
          v-model="newListName"
          type="text"
          maxlength="200"
          placeholder="e.g. Shopping List"
          required
        />
      </div>
      <div class="form-actions">
        <button type="button" class="btn btn-secondary" @click="showNew = false">Cancel</button>
        <button type="submit" class="btn btn-primary">Create</button>
      </div>
    </form>
  </BrainModal>

  <BrainModal v-model="showRename" size="sm">
    <div class="modal-header">
      <h3>Rename List</h3>
      <button class="btn-close" @click="showRename = false">
        <X :size="16" />
      </button>
    </div>
    <form @submit.prevent="renameList">
      <div class="form-group">
        <label for="renameInput">New Name</label>
        <input id="renameInput" v-model="renameName" type="text" maxlength="200" required />
      </div>
      <div class="form-actions">
        <button type="button" class="btn btn-secondary" @click="showRename = false">Cancel</button>
        <button type="submit" class="btn btn-primary">Rename</button>
      </div>
    </form>
  </BrainModal>
</template>
