<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import type { ScheduleInput, ScheduleItem } from '../../api/scheduler';
import { scheduler } from '../../api/scheduler';
import { formatDate } from '../../utils/format';
import { apiErrorMessage } from '../../api/http';
import { useToast } from '../../composables/useToast';
import { useConfirm } from '../../composables/useConfirm';
import { useBrainResource } from '../../composables/useBrainResource';
import ToggleSwitch from '../../ui/ToggleSwitch.vue';
import { Calendar, ChevronLeft, Plus } from '@lucide/vue';

const { show: showToast } = useToast();
const { confirm } = useConfirm();

const {
  data: items,
  loading,
  reload: load,
} = useBrainResource(async () => await scheduler.list(), {
  initial: [] as ScheduleItem[],
  failMsg: 'Failed to load schedules',
});

const formMode = ref<'list' | 'form'>('list');
const editingId = ref<number | null>(null);
const formMsg = ref('');
const formStart = ref('');
const formDay = ref<number | null>(null);
const formHour = ref<number | null>(null);
const formMinute = ref<number | null>(null);
const formEnabled = ref(true);

const days = Array.from({ length: 31 }, (_, i) => i + 1);
const hours = Array.from({ length: 24 }, (_, i) => i);
const minutes = Array.from({ length: 60 }, (_, i) => i);

function pad(n: number): string {
  return String(n).padStart(2, '0');
}

// A field may only be "Every" if every finer field is also "Every" — the
// submittable shapes are EEE / EEF / EFF / FFF (E=Every, F=Fixed, ordered
// day > hour > minute). Disable the "Every" option on a finer selector once
// a coarser one is fixed, and auto-correct now-illegal values below.
const hourEveryDisabled = computed(() => formDay.value !== null);
const minuteEveryDisabled = computed(() => formHour.value !== null);

function enforceEveryPrefix(): void {
  if (formDay.value !== null && formHour.value === null) formHour.value = 0;
  if (formHour.value !== null && formMinute.value === null) formMinute.value = 0;
}

watch(formDay, enforceEveryPrefix);
watch(formHour, enforceEveryPrefix);

function cadenceLabel(s: ScheduleItem): string {
  const { day, hour, minute } = s;
  if (day == null && hour == null && minute == null) return 'Every minute';
  if (day == null && hour == null) return `Every hour at :${pad(minute as number)}`;
  if (day == null) return `Every day at ${pad(hour as number)}:${pad(minute as number)}`;
  return `Monthly on day ${day} at ${pad(hour as number)}:${pad(minute as number)}`;
}

function openForm(item: ScheduleItem | null): void {
  formMsg.value = item?.message ?? item?.prompt ?? '';
  formStart.value = item?.start_at ? item.start_at.slice(0, 16) : '';
  formDay.value = item?.day ?? null;
  formHour.value = item?.hour ?? null;
  formMinute.value = item?.minute ?? null;
  formEnabled.value = item?.enabled !== 0;
  enforceEveryPrefix();
  editingId.value = item?.id ?? null;
  formMode.value = 'form';
}

async function save(): Promise<void> {
  const body: ScheduleInput = {
    message: formMsg.value.trim(),
    day: formDay.value,
    hour: formHour.value,
    minute: formMinute.value,
    enabled: formEnabled.value,
  };
  if (formStart.value) body.start_at = formStart.value;
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

// Quick-toggle from the list: reuse the update path with just the enabled flag
// flipped. start_at/cron fields are left untouched — the poller re-evaluates
// them against the wall clock on every cycle, so re-enabling simply lets the
// next matching minute fire again (no recompute needed).
async function toggleEnabled(s: ScheduleItem, next: boolean): Promise<void> {
  try {
    await scheduler.update(s.id, {
      message: s.message || s.prompt || '',
      day: s.day,
      hour: s.hour,
      minute: s.minute,
      enabled: next,
    });
    showToast(next ? 'Schedule enabled' : 'Schedule disabled', 'success');
    await load();
  } catch (e) {
    showToast(apiErrorMessage(e, 'Toggle failed'), 'error');
  }
}

// Delete = hard delete — the row (and its id) is gone for good, so its
// schedule thread can never be re-entered. Distinct from Disable, which pauses
// firing but keeps the row (and the id reusable for re-enabling).
async function cancelSchedule(s: ScheduleItem): Promise<void> {
  const ok = await confirm({
    title: 'Delete Schedule',
    desc: 'Delete this schedule? To pause it without deleting, use the toggle instead.',
    confirmLabel: 'Delete',
    confirmClass: 'btn-danger',
  });
  if (!ok) return;
  try {
    await scheduler.delete(s.id);
    showToast('Schedule deleted', 'success');
    await load();
  } catch (e) {
    showToast(apiErrorMessage(e, 'Delete failed'), 'error');
  }
}
</script>

<template>
  <div class="panel-header">
    <h2>Scheduler</h2>
    <button class="btn btn-primary" @click="openForm(null)">
      <Plus :size="14" /> New Schedule
    </button>
  </div>

  <div v-if="loading" class="loading">Loading…</div>

  <div v-else-if="formMode === 'form'" class="provider-form-page">
    <div class="form-page-header">
      <button class="btn btn-secondary btn-sm back-btn" @click="formMode = 'list'">
        <ChevronLeft :size="14" /> Back
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
        <label for="schedStart">Start Time</label>
        <input id="schedStart" v-model="formStart" type="datetime-local" />
      </div>
      <div class="form-group">
        <label for="schedDay">Day of month</label>
        <select id="schedDay" v-model="formDay">
          <option :value="null">Every</option>
          <option v-for="d in days" :key="d" :value="d">{{ d }}</option>
        </select>
      </div>
      <div class="form-group">
        <label for="schedHour">Hour</label>
        <select id="schedHour" v-model="formHour">
          <option :value="null" :disabled="hourEveryDisabled">Every</option>
          <option v-for="h in hours" :key="h" :value="h">{{ pad(h) }}</option>
        </select>
      </div>
      <div class="form-group">
        <label for="schedMinute">Minute</label>
        <select id="schedMinute" v-model="formMinute">
          <option :value="null" :disabled="minuteEveryDisabled">Every</option>
          <option v-for="m in minutes" :key="m" :value="m">{{ pad(m) }}</option>
        </select>
      </div>
      <div class="form-group">
        <label>Enabled</label>
        <ToggleSwitch v-model="formEnabled" />
        <p class="form-hint">Disable to pause firing without deleting the schedule.</p>
      </div>
      <div class="form-actions">
        <button type="button" class="btn btn-secondary" @click="formMode = 'list'">Cancel</button>
        <button type="submit" class="btn btn-primary">Save</button>
      </div>
    </form>
  </div>

  <div v-else-if="items.length === 0" class="empty-state">
    <div class="empty-icon">
      <Calendar :size="40" />
    </div>
    <h3>No schedules</h3>
    <p>Create your first scheduled task.</p>
  </div>

  <table v-else class="records-table">
    <thead>
      <tr>
        <th>Message</th>
        <th>State</th>
        <th>Start</th>
        <th>Cadence</th>
        <th></th>
      </tr>
    </thead>
    <tbody>
      <tr v-for="s in items" :key="s.id">
        <td class="key-cell">{{ s.message || s.prompt || '' }}</td>
        <td class="state-cell">
          <ToggleSwitch
            :model-value="s.enabled === 1"
            @update:model-value="toggleEnabled(s, $event)"
          />
          <span :class="s.enabled === 1 ? 'state-active' : 'state-disabled'">
            {{ s.enabled === 1 ? 'Active' : 'Disabled' }}
          </span>
        </td>
        <td>{{ formatDate(s.start_at) }}</td>
        <td>{{ cadenceLabel(s) }}</td>
        <td class="row-actions">
          <button class="btn btn-sm btn-secondary" @click="openForm(s)">Edit</button>
          <button class="btn btn-sm btn-danger" @click="cancelSchedule(s)">Delete</button>
        </td>
      </tr>
    </tbody>
  </table>
</template>

<style scoped>
.state-cell {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.state-active {
  color: var(--success);
}
.state-disabled {
  color: var(--text-muted);
}
</style>
