<script setup lang="ts">
import { computed, ref } from 'vue';
import { describeCron } from '@chalie/shared';
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
// Crontab fields — free-text expressions (`*`, `N`, `N-M`, `*/S`, comma-lists);
// `*` = every. There is no every-prefix invariant any more, so no cross-field
// coupling or auto-correction: each field stands alone and the backend
// validates the full 5-field shape on save.
const formMinute = ref('*');
const formHour = ref('*');
const formDay = ref('*');
const formMonth = ref('*');
const formWeekday = ref('*');
const formEnabled = ref(true);

// Live human-readable preview as the user edits — the same describeCron the
// interface uses, so a schedule reads identically on both surfaces.
const cronPreview = computed(() =>
  describeCron(formMinute.value, formHour.value, formDay.value, formMonth.value, formWeekday.value),
);

function cadenceLabel(s: ScheduleItem): string {
  return describeCron(s.minute, s.hour, s.day, s.month, s.weekday);
}

function openForm(item: ScheduleItem | null): void {
  formMsg.value = item?.message ?? item?.prompt ?? '';
  formStart.value = item?.start_at ? item.start_at.slice(0, 16) : '';
  formMinute.value = item?.minute ?? '*';
  formHour.value = item?.hour ?? '*';
  formDay.value = item?.day ?? '*';
  formMonth.value = item?.month ?? '*';
  formWeekday.value = item?.weekday ?? '*';
  formEnabled.value = item?.enabled !== 0;
  editingId.value = item?.id ?? null;
  formMode.value = 'form';
}

// A blank field is normalized to '*' so an empty input reads as "every" rather
// than failing validation; the backend is the sole authority on cron validity.
function cronField(v: string): string {
  return v.trim() || '*';
}

async function save(): Promise<void> {
  const body: ScheduleInput = {
    message: formMsg.value.trim(),
    minute: cronField(formMinute.value),
    hour: cronField(formHour.value),
    day: cronField(formDay.value),
    month: cronField(formMonth.value),
    weekday: cronField(formWeekday.value),
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
      minute: s.minute,
      hour: s.hour,
      day: s.day,
      month: s.month,
      weekday: s.weekday,
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
        <label>Recurrence (crontab)</label>
        <div class="cron-grid">
          <div class="cron-field">
            <label for="cronMinute">Minute</label>
            <input id="cronMinute" v-model="formMinute" placeholder="*" spellcheck="false" />
          </div>
          <div class="cron-field">
            <label for="cronHour">Hour</label>
            <input id="cronHour" v-model="formHour" placeholder="*" spellcheck="false" />
          </div>
          <div class="cron-field">
            <label for="cronDay">Day</label>
            <input id="cronDay" v-model="formDay" placeholder="*" spellcheck="false" />
          </div>
          <div class="cron-field">
            <label for="cronMonth">Month</label>
            <input id="cronMonth" v-model="formMonth" placeholder="*" spellcheck="false" />
          </div>
          <div class="cron-field">
            <label for="cronWeekday">Weekday</label>
            <input id="cronWeekday" v-model="formWeekday" placeholder="*" spellcheck="false" />
          </div>
        </div>
        <p class="cron-preview">{{ cronPreview }}</p>
        <p class="form-hint">
          Standard crontab: <code>*</code> = every, <code>*/5</code> = every 5th,
          <code>1-5</code> = range, <code>1,15</code> = list. Weekday 0–6 (0 = Sunday).
        </p>
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

/* Crontab field grid — five compact expression inputs on one row. */
.cron-grid {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 0.5rem;
}
.cron-field {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}
.cron-field label {
  font-size: 0.72rem;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.cron-field input {
  font-family: var(--font-mono, ui-monospace, monospace);
  text-align: center;
}
.cron-preview {
  margin: 0.5rem 0 0.25rem;
  font-weight: 500;
  color: var(--text-primary);
}
</style>
