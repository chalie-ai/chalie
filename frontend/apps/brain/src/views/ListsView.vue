<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue';
import { lists as listsApi } from '../api/lists';
import type { List, ListItem } from '../api/lists';
import { useToast } from '../composables/useToast';
import { useConfirm } from '../composables/useConfirm';
import BrainModal from '../ui/BrainModal.vue';
import BrainIcon from '../ui/BrainIcon.vue';
import { HttpError } from '@chalie/shared';

const { show: showToast } = useToast();
const { confirm } = useConfirm();

const listsData = ref<List[]>([]);
const loading = ref(true);
const expanded = reactive<Record<string, boolean>>({});

// Modal state
const showNew = ref(false);
const newListName = ref('');
const showRename = ref(false);
const renameName = ref('');
const renameId = ref<string | number>('');

// --- counts helper (legacy lists.js:48-51) ---
function counts(list: List): { done: number; total: number; pct: number } {
  const items = list.items ?? [];
  const done = list.items ? items.filter((i) => i.checked).length : (list.checked_count ?? 0);
  const total = list.items ? items.length : (list.item_count ?? 0);
  const pct = total > 0 ? Math.round((done / total) * 100) : 0;
  return { done, total, pct };
}

// --- load (legacy lists.js:28-37) ---
async function load(): Promise<void> {
  loading.value = true;
  try {
    const d = await listsApi.list();
    listsData.value = d.items ?? [];
  } catch {
    showToast('Failed to load lists', 'error');
  } finally {
    loading.value = false;
  }
}

// --- fetchListDetail (legacy lists.js:113-123) ---
async function fetchListDetail(list: List): Promise<void> {
  try {
    const d = await listsApi.get(list.id);
    list.items = d.item.items ?? [];
  } catch {
    showToast('Failed to load list items', 'error');
  }
}

// --- toggle expand/collapse (legacy lists.js:79-87) ---
function toggle(list: List): void {
  const id = String(list.id);
  expanded[id] = !expanded[id];
  if (expanded[id]) fetchListDetail(list);
}

// --- toggleItem (legacy lists.js:125-136) ---
async function toggleItem(list: List, item: ListItem, checked: boolean): Promise<void> {
  const endpoint = checked ? 'check' : 'uncheck';
  try {
    await listsApi.toggleItem(list.id, endpoint, { content: item.content });
    item.checked = checked;
  } catch {
    showToast('Failed to update item', 'error');
  }
}

// --- onAddKey handler (legacy lists.js:96-103) ---
function onAddKey(e: KeyboardEvent, list: List): void {
  const input = e.target as HTMLInputElement;
  const value = input.value.trim();
  if (value) {
    addItem(list, value);
    input.value = '';
  }
}

// --- addItem (legacy lists.js:138-143) ---
async function addItem(list: List, text: string): Promise<void> {
  try {
    await listsApi.addItems(list.id, [text]);
    await fetchListDetail(list);
  } catch (e) {
    // mirrors legacy lists.js:138-143 — silent on non-ok, toast on network error
    if (!(e instanceof HttpError)) showToast('Failed to add item', 'error');
  }
}

// --- openRename (legacy lists.js:169-192) ---
function openRename(list: List): void {
  renameId.value = list.id;
  renameName.value = list.name;
  showRename.value = true;
}

// --- createList (legacy lists.js:159-166) ---
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

// --- renameList (legacy lists.js:184-191) ---
async function renameList(): Promise<void> {
  try {
    await listsApi.rename(renameId.value, renameName.value.trim());
    showRename.value = false;
    showToast('List renamed', 'success');
    await load();
  } catch (e) {
    showToast(e instanceof HttpError ? 'Failed to rename' : 'Network error', 'error');
  }
}

// --- deleteList (legacy lists.js:194-213) ---
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

onMounted(load);
</script>

<template>
  <!-- Header (legacy lists.js:17-20) -->
  <div class="panel-header">
    <h2>Lists</h2>
    <button class="btn btn-primary" @click="showNew = true; newListName = ''">
      <BrainIcon name="Plus" :size="14" /> New List
    </button>
  </div>

  <!-- Loading state -->
  <div v-if="loading" class="loading">Loading…</div>

  <!-- Empty state (legacy lists.js:43, VisionView inline pattern) -->
  <div v-else-if="listsData.length === 0" class="empty-state">
    <div class="empty-icon">
      <BrainIcon name="List" :size="40" />
    </div>
    <h3>No lists</h3>
    <p>Create your first list to get started.</p>
  </div>

  <!-- List cards (legacy lists.js:47-77) -->
  <template v-else>
    <div v-for="list in listsData" :key="list.id" class="list-card">
      <div class="list-card-header" @click="toggle(list)">
        <div class="list-card-title">
          <span class="list-chev">
            <BrainIcon :name="expanded[String(list.id)] ? 'ChevronDown' : 'Chevron'" :size="14" />
          </span>
          <span>{{ list.name }}</span>
          <span class="list-count">{{ counts(list).done }}/{{ counts(list).total }}</span>
        </div>
        <div class="list-card-actions">
          <button class="btn btn-sm btn-secondary" @click.stop="openRename(list)">Rename</button>
          <button class="btn btn-sm btn-danger" @click.stop="deleteList(list)">Delete</button>
        </div>
      </div>

      <div v-if="counts(list).total > 0" class="progress-bar">
        <div class="progress-fill" :style="{ width: counts(list).pct + '%' }"></div>
      </div>

      <div v-if="expanded[String(list.id)]" class="list-items">
        <label
          v-for="item in list.items ?? []"
          :key="item.id"
          class="list-item"
        >
          <input
            type="checkbox"
            :checked="item.checked"
            @change="toggleItem(list, item, ($event.target as HTMLInputElement).checked)"
          >
          <span :class="{ done: item.checked }">{{ item.content }}</span>
        </label>
        <div class="list-add-row">
          <input
            type="text"
            class="list-add-input"
            placeholder="Add item…"
            maxlength="300"
            @keydown.enter="onAddKey($event, list)"
          >
        </div>
      </div>
    </div>
  </template>

  <!-- Create list modal (legacy lists.js:145-167) -->
  <BrainModal v-model="showNew" size="sm">
    <div class="modal-header">
      <h3>New List</h3>
      <button class="btn-close" @click="showNew = false">
        <BrainIcon name="Close" :size="16" />
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
        >
      </div>
      <div class="form-actions">
        <button type="button" class="btn btn-secondary" @click="showNew = false">Cancel</button>
        <button type="submit" class="btn btn-primary">Create</button>
      </div>
    </form>
  </BrainModal>

  <!-- Rename list modal (legacy lists.js:169-192) -->
  <BrainModal v-model="showRename" size="sm">
    <div class="modal-header">
      <h3>Rename List</h3>
      <button class="btn-close" @click="showRename = false">
        <BrainIcon name="Close" :size="16" />
      </button>
    </div>
    <form @submit.prevent="renameList">
      <div class="form-group">
        <label for="renameInput">New Name</label>
        <input
          id="renameInput"
          v-model="renameName"
          type="text"
          maxlength="200"
          required
        >
      </div>
      <div class="form-actions">
        <button type="button" class="btn btn-secondary" @click="showRename = false">Cancel</button>
        <button type="submit" class="btn btn-primary">Rename</button>
      </div>
    </form>
  </BrainModal>
</template>
