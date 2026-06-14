<script setup lang="ts">
import { ref, computed, onMounted } from 'vue';
import { scheduler } from '../../api/scheduler';
import type { ScheduleItem, ScheduleInput } from '../../api/scheduler';
import { formatDate } from '../../utils/format';
import { apiErrorMessage } from '../../api/http';
import { useToast } from '../../composables/useToast';
import { useConfirm } from '../../composables/useConfirm';
import BrainIcon from '../../ui/BrainIcon.vue';

const props = defineProps<{ statusFilter: 'all' | 'pending' | 'fired' | 'failed' | 'cancelled' }>();

const { show: showToast } = useToast();
const { confirm } = useConfirm();

const items = ref<ScheduleItem[]>([]);
const loading = ref(true);

// Inline form state
type FormMode = 'list' | 'form';
const formMode = ref<FormMode>('list');
const editingId = ref<string | number | null>(null);
const formMsg = ref('');
const formDue = ref('');
const formType = ref<'notification' | 'prompt'>('notification');
const formRecur = ref('');

const filtered = computed<ScheduleItem[]>(() =>
  props.statusFilter === 'all' ? items.value : items.value.filter((s) => s.status === props.statusFilter),
);

function statusClass(status: string | null | undefined): string {
  const map: Record<string, string> = {
    pending: 'badge-warning',
    fired: 'badge-success',
    failed: 'badge-danger',
    cancelled: 'badge-muted',
  };
  return map[status ?? ''] ?? 'badge-muted';
}

async function load(): Promise<void> {
  loading.value = true;
  try {
    const data = await scheduler.list();
    items.value = data.schedules ?? data.items ?? [];
  } catch {
    showToast('Failed to load schedules', 'error');
  } finally {
    loading.value = false;
  }
}

function openForm(item: ScheduleItem | null): void {
  formMsg.value = item?.message ?? item?.prompt ?? '';
  formDue.value = item?.due_at ? item.due_at.slice(0, 16) : '';
  formType.value = item?.type === 'prompt' || item?.item_type === 'prompt' ? 'prompt' : 'notification';
  formRecur.value = item?.recurrence ?? '';
  editingId.value = item?.id ?? null;
  formMode.value = 'form';
}

async function save(): Promise<void> {
  const body: ScheduleInput & { recurrence?: string | null } = {
    message: formMsg.value.trim(),
    due_at: formDue.value,
    type: formType.value,
    recurrence: formRecur.value || null,
  };
  try {
    if (editingId.value != null) {
      await scheduler.update(editingId.value, body);
    } else {
      await scheduler.create(body);
    }
    showToast(editingId.value != null ? 'Schedule updated' : 'Schedule created', 'success');
    formMode.value = 'list';
    await load();
  } catch (e) {
    showToast(apiErrorMessage(e, 'Save failed'), 'error');
  }
}

async function cancelSchedule(s: ScheduleItem): Promise<void> {
  const ok = await confirm({
    title: 'Cancel Schedule',
    desc: 'Cancel this scheduled item?',
    confirmLabel: 'Cancel Item',
    confirmClass: 'btn-danger',
  });
  if (!ok) return;
  try {
    await scheduler.delete(s.id);
    showToast('Schedule cancelled', 'success');
    await load();
  } catch (e) {
    showToast(apiErrorMessage(e, 'Cancel failed'), 'error');
  }
}

onMounted(load);
</script>

<template>
  <div class="panel-header">
    <h2>Scheduler</h2>
    <button class="btn btn-primary" @click="openForm(null)">
      <BrainIcon name="Plus" :size="14" /> New Schedule
    </button>
  </div>

  <div v-if="loading" class="loading">Loading…</div>

  <template v-else-if="formMode === 'form'">
    <div class="provider-form-page">
      <div class="form-page-header">
        <button class="btn btn-secondary btn-sm back-btn" @click="formMode = 'list'">
          <BrainIcon name="Chevron" :size="14" /> Back
        </button>
        <h3>{{ editingId != null ? 'Edit Schedule' : 'New Schedule' }}</h3>
      </div>
      <form @submit.prevent="save">
        <div class="form-group">
          <label for="schedMsg">Prompt / Message</label>
          <textarea
            id="schedMsg"
            v-model="formMsg"
            rows="3"
            maxlength="1000"
            placeholder="What Chalie should do when this fires"
            required
          ></textarea>
        </div>
        <div class="form-group">
          <label for="schedDue">Due Date &amp; Time</label>
          <input id="schedDue" v-model="formDue" type="datetime-local" required>
        </div>
        <div class="form-group">
          <label for="schedType">Type</label>
          <select id="schedType" v-model="formType">
            <option value="notification">Notification</option>
            <option value="prompt">Prompt</option>
          </select>
        </div>
        <div class="form-group">
          <label for="schedRecur">Recurrence</label>
          <select id="schedRecur" v-model="formRecur">
            <option value="">None (one-time)</option>
            <option value="interval">Every X minutes</option>
            <option value="hourly">Hourly</option>
            <option value="daily">Daily</option>
            <option value="weekdays">Weekdays</option>
            <option value="weekly">Weekly</option>
            <option value="monthly">Monthly</option>
          </select>
        </div>
        <div class="form-actions">
          <button type="button" class="btn btn-secondary" @click="formMode = 'list'">Cancel</button>
          <button type="submit" class="btn btn-primary">Save</button>
        </div>
      </form>
    </div>
  </template>

  <template v-else-if="filtered.length === 0">
    <div class="empty-state">
      <div class="empty-icon">
        <BrainIcon name="Calendar" :size="40" />
      </div>
      <h3>No schedules</h3>
      <p>Create your first scheduled task.</p>
    </div>
  </template>

  <template v-else>
    <table class="records-table">
      <thead>
        <tr>
          <th>Message</th>
          <th>Status</th>
          <th>Due</th>
          <th>Recurrence</th>
          <th>Type</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="s in filtered" :key="s.id">
          <td class="key-cell">{{ s.message || s.prompt || '' }}</td>
          <td><span class="badge" :class="statusClass(s.status)">{{ s.status || '' }}</span></td>
          <td>{{ formatDate(s.due_at || s.due) }}</td>
          <td>{{ s.recurrence || '—' }}</td>
          <td><span class="badge badge-muted">{{ s.type || 'notification' }}</span></td>
          <td class="row-actions">
            <button class="btn btn-sm btn-secondary" @click="openForm(s)">Edit</button>
            <button v-if="s.status === 'pending'" class="btn btn-sm btn-danger" @click="cancelSchedule(s)">Cancel</button>
          </td>
        </tr>
      </tbody>
    </table>
  </template>
</template>
