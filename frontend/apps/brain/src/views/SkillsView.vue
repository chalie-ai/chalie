<script setup lang="ts">
import { ref, computed, onMounted } from 'vue';
import { skills as skillsApi } from '../api/skills';
import type { Skill, Association } from '../api/skills';
import { formatDate } from '../utils/format';
import { apiErrorMessage } from '../api/http';
import { useToast } from '../composables/useToast';
import { useConfirm } from '../composables/useConfirm';
import BrainIcon from '../ui/BrainIcon.vue';

const { show: showToast } = useToast();
const { confirm } = useConfirm();

const skills = ref<Skill[]>([]);
const associations = ref<Association[]>([]);
const loading = ref(true);
const expandedId = ref<string | number | null>(null);
const editingId = ref<string | number | null>(null);
const viewMode = ref<'list' | 'create'>('list');

const userSkills = computed<Skill[]>(() => skills.value.filter((s) => s.source === 'user'));
const curatedSkills = computed<Skill[]>(() => skills.value.filter((s) => s.source === 'curated'));

// Create-form refs
const createTitle = ref('');
const createUseFor = ref('');
const createTags = ref('');
const createContent = ref('');

// Edit-form refs (only one card is ever in edit mode at a time)
const editUseFor = ref('');
const editTags = ref('');
const editContent = ref('');

// ── Load ─────────────────────────────────────────────────────────────

async function load(): Promise<void> {
  loading.value = true;
  try {
    const data = await skillsApi.list();
    skills.value = data.skills ?? [];
    associations.value = data.associations ?? [];
  } catch {
    showToast('Failed to load skills', 'error');
  } finally {
    loading.value = false;
  }
}

// ── Expand ───────────────────────────────────────────────────────────

function toggleExpand(skill: Skill): void {
  expandedId.value = expandedId.value === skill.id ? null : skill.id;
}

// ── Edit ─────────────────────────────────────────────────────────────

function startEdit(skill: Skill): void {
  editUseFor.value = skill.use_for;
  editTags.value = skill.tags ?? '';
  editContent.value = skill.content;
  editingId.value = skill.id;
}

async function saveEdit(skill: Skill): Promise<void> {
  const use_for = editUseFor.value.trim();
  const content = editContent.value.trim();
  const tags = editTags.value.trim();
  try {
    const data = await skillsApi.update(skill.id, { use_for, content, tags });
    showToast('Skill updated', 'success');
    const idx = skills.value.findIndex((s) => s.id === skill.id);
    if (idx !== -1) skills.value[idx] = data.skill;
    editingId.value = null;
  } catch (e) {
    showToast(apiErrorMessage(e, 'Failed to update skill'), 'error');
  }
}

// ── Toggle ───────────────────────────────────────────────────────────

async function toggleSkill(skill: Skill): Promise<void> {
  try {
    const data = await skillsApi.toggle(skill.id);
    skill.enabled = data.enabled;
    showToast(data.enabled ? 'Skill enabled' : 'Skill disabled', 'success');
  } catch (e) {
    showToast(apiErrorMessage(e, 'Failed to toggle skill'), 'error');
  }
}

// ── Delete ───────────────────────────────────────────────────────────

async function deleteSkill(skill: Skill): Promise<void> {
  const ok = await confirm({
    title: 'Delete Skill',
    desc: `Delete "${skill.title}"? This cannot be undone.`,
    confirmLabel: 'Delete',
    confirmClass: 'btn-danger',
  });
  if (!ok) return;
  try {
    await skillsApi.delete(skill.id);
    showToast('Skill deleted', 'success');
    skills.value = skills.value.filter((s) => s.id !== skill.id);
    editingId.value = null;
  } catch (e) {
    showToast(apiErrorMessage(e, 'Failed to delete skill'), 'error');
  }
}

// ── Copy ─────────────────────────────────────────────────────────────

async function copySkill(skill: Skill): Promise<void> {
  const ok = await confirm({
    title: 'Customise Skill',
    desc: `Copy "${skill.title}" as a customisable skill? The curated version will be disabled.`,
    confirmLabel: 'Customise',
    confirmClass: 'btn-primary',
  });
  if (!ok) return;
  try {
    const data = await skillsApi.copy(skill.id);
    showToast(`Skill copied as "${data.skill.title}"`, 'success');
    await load();
  } catch (e) {
    showToast(apiErrorMessage(e, 'Failed to copy skill'), 'error');
  }
}

// ── Create ───────────────────────────────────────────────────────────

function openCreate(): void {
  createTitle.value = '';
  createUseFor.value = '';
  createTags.value = '';
  createContent.value = '';
  viewMode.value = 'create';
}

async function submitCreate(): Promise<void> {
  const title = createTitle.value.trim();
  const use_for = createUseFor.value.trim();
  const content = createContent.value.trim();
  const tags = createTags.value.trim();
  try {
    const data = await skillsApi.create({ title, use_for, content, tags });
    showToast(`Skill "${title}" created`, 'success');
    skills.value.push(data.skill);
    editingId.value = null;
    viewMode.value = 'list';
  } catch (e) {
    showToast(apiErrorMessage(e, 'Failed to create skill'), 'error');
  }
}

onMounted(load);
</script>

<template>
  <!-- Header -->
  <div class="panel-header">
    <h2><BrainIcon name="Skill" :size="20" /> Skills</h2>
    <div class="panel-header-actions">
      <button class="btn btn-primary btn-sm" @click="openCreate">
        <BrainIcon name="Plus" :size="14" /> New Skill
      </button>
    </div>
  </div>

  <div v-if="loading" class="loading">Loading…</div>

  <!-- Create form -->
  <template v-else-if="viewMode === 'create'">
    <div class="provider-form-page">
      <div class="form-page-header">
        <button class="btn btn-secondary btn-sm" @click="viewMode = 'list'">
          <BrainIcon name="Chevron" :size="14" /> Back
        </button>
        <h3>New Skill</h3>
      </div>
      <form @submit.prevent="submitCreate">
        <div class="form-group">
          <label>Title <span style="color:var(--error)">*</span></label>
          <input
            v-model="createTitle"
            type="text"
            placeholder="e.g. Track Package Delivery"
            required
          >
        </div>
        <div class="form-group">
          <label>Use for <span style="color:var(--error)">*</span></label>
          <input
            v-model="createUseFor"
            type="text"
            placeholder="One sentence: when should this skill be used?"
            required
          >
        </div>
        <div class="form-group">
          <label>Tags</label>
          <input
            v-model="createTags"
            type="text"
            placeholder="logistics, tracking, delivery"
          >
        </div>
        <div class="form-group">
          <label>Instructions <span style="color:var(--error)">*</span></label>
          <textarea
            v-model="createContent"
            rows="10"
            :placeholder="'1. First step (reference tools like `search`, `memory`)\n2. Second step\n3. Third step'"
            required
          ></textarea>
        </div>
        <div class="form-actions">
          <button type="button" class="btn btn-secondary" @click="viewMode = 'list'">Cancel</button>
          <button type="submit" class="btn btn-primary">Create Skill</button>
        </div>
      </form>
    </div>
  </template>

  <!-- List view -->
  <template v-else>
    <!-- My Skills -->
    <h4 class="section-head">My Skills</h4>

    <div
      v-if="userSkills.length === 0"
      class="empty-state"
      style="margin-bottom:24px;"
    >
      <div class="empty-icon">
        <BrainIcon name="Skill" :size="40" />
      </div>
      <h3>No custom skills yet</h3>
      <p>Click "New Skill" to create a step-by-step playbook for a recurring task.</p>
    </div>

    <div v-else id="userSkillsGrid" class="skills-grid">
      <template v-for="skill in userSkills" :key="skill.id">
        <!-- Edit mode -->
        <div v-if="editingId === skill.id" class="cap-card skill-card">
          <div class="skill-card-header">
            <strong>{{ skill.title }}</strong>
            <span class="badge badge-violet">v{{ skill.version }}</span>
          </div>
          <form class="skill-edit-form" @submit.prevent="saveEdit(skill)">
            <div class="form-group">
              <label>Use for</label>
              <input
                v-model="editUseFor"
                type="text"
                class="skill-field-use_for"
                placeholder="When should this skill be used?"
              >
            </div>
            <div class="form-group">
              <label>Tags</label>
              <input
                v-model="editTags"
                type="text"
                class="skill-field-tags"
                placeholder="tag1, tag2"
              >
            </div>
            <div class="form-group">
              <label>Instructions</label>
              <textarea
                v-model="editContent"
                class="skill-field-content"
                rows="8"
                :placeholder="'1. First step\n2. Second step'"
              ></textarea>
            </div>
            <div class="form-actions">
              <button type="button" class="btn btn-secondary btn-sm" @click="editingId = null">Cancel</button>
              <button type="submit" class="btn btn-primary btn-sm">Save</button>
            </div>
          </form>
        </div>

        <!-- Display mode -->
        <div
          v-else
          class="cap-card skill-card"
          :class="{ 'skill-expanded': expandedId === skill.id }"
        >
          <div class="skill-card-header">
            <div class="skill-card-title" style="cursor:pointer;" @click="toggleExpand(skill)">
              <span class="skill-expand-icon">
                <BrainIcon :name="expandedId === skill.id ? 'ChevronDown' : 'Chevron'" :size="12" />
              </span>
              <strong>{{ skill.title }}</strong>
              <span class="badge badge-violet">v{{ skill.version }}</span>
              <span v-if="skill.enabled" class="badge badge-success">enabled</span>
              <span v-else class="badge badge-muted">disabled</span>
            </div>
            <div class="skill-card-actions">
              <label
                class="switch-label skill-toggle-wrap"
                :title="skill.enabled ? 'Disable' : 'Enable'"
              >
                <label class="switch">
                  <input
                    type="checkbox"
                    class="skill-toggle"
                    :checked="skill.enabled"
                    @change="toggleSkill(skill)"
                  >
                  <span class="switch-track"></span>
                </label>
              </label>
              <button class="btn btn-secondary btn-sm" @click="startEdit(skill)">
                <BrainIcon name="Edit" :size="13" />
              </button>
              <button class="btn btn-danger btn-sm" @click="deleteSkill(skill)">
                <BrainIcon name="Trash" :size="13" />
              </button>
            </div>
          </div>
          <div class="skill-use-for">{{ skill.use_for }}</div>
          <div v-if="skill.tags" class="skill-tags">
            <span
              v-for="tag in skill.tags.split(',').map(t => t.trim()).filter(Boolean)"
              :key="tag"
              class="badge badge-muted skill-tag"
            >{{ tag }}</span>
          </div>
          <div v-if="expandedId === skill.id" class="skill-expanded-content">
            <pre class="skill-content-preview">{{ skill.content }}</pre>
          </div>
        </div>
      </template>
    </div>

    <!-- Curated Skills -->
    <h4 class="section-head" style="margin-top:32px;">Curated Skills</h4>

    <div v-if="curatedSkills.length === 0" class="empty-state">
      <p>No curated skills loaded.</p>
    </div>

    <div v-else id="curatedSkillsGrid" class="skills-grid">
      <div
        v-for="skill in curatedSkills"
        :key="skill.id"
        class="cap-card skill-card"
        :class="{ 'skill-disabled': !skill.enabled, 'skill-expanded': expandedId === skill.id }"
      >
        <div class="skill-card-header">
          <div class="skill-card-title" style="cursor:pointer;" @click="toggleExpand(skill)">
            <span class="skill-expand-icon">
              <BrainIcon :name="expandedId === skill.id ? 'ChevronDown' : 'Chevron'" :size="12" />
            </span>
            <strong>{{ skill.title }}</strong>
            <span class="badge badge-muted">v{{ skill.version }}</span>
            <span v-if="skill.enabled" class="badge badge-success">enabled</span>
            <span v-else class="badge badge-muted">disabled</span>
          </div>
          <div class="skill-card-actions">
            <label
              class="switch-label skill-toggle-wrap"
              :title="skill.enabled ? 'Disable' : 'Enable'"
            >
              <label class="switch">
                <input
                  type="checkbox"
                  class="skill-toggle"
                  :checked="skill.enabled"
                  @change="toggleSkill(skill)"
                >
                <span class="switch-track"></span>
              </label>
            </label>
            <button
              class="btn btn-secondary btn-sm"
              title="Copy &amp; Customise"
              @click="copySkill(skill)"
            >
              <BrainIcon name="Copy" :size="13" /> Customise
            </button>
          </div>
        </div>
        <div class="skill-use-for">{{ skill.use_for }}</div>
        <div v-if="skill.tags" class="skill-tags">
          <span
            v-for="tag in skill.tags.split(',').map(t => t.trim()).filter(Boolean)"
            :key="tag"
            class="badge badge-muted skill-tag"
          >{{ tag }}</span>
        </div>
        <div v-if="expandedId === skill.id" class="skill-expanded-content">
          <pre class="skill-content-preview">{{ skill.content }}</pre>
        </div>
      </div>
    </div>

    <!-- Skill Associations -->
    <template v-if="associations.length > 0">
      <h4 class="section-head" style="margin-top:32px;">Skill Associations</h4>
      <p class="panel-desc">Patterns discovered from your behaviour, linked to skills.</p>
      <table class="records-table">
        <thead>
          <tr>
            <th>Pattern</th>
            <th>Skill</th>
            <th>Rule</th>
            <th>Since</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(a, idx) in associations" :key="idx">
            <td><span class="badge badge-violet">{{ a.pattern_name }}</span></td>
            <td>{{ a.skill_title }}</td>
            <td style="font-size:12px;color:var(--text-secondary);">{{ a.rule }}</td>
            <td style="font-size:12px;color:var(--text-tertiary);">{{ formatDate(a.created_at) }}</td>
          </tr>
        </tbody>
      </table>
    </template>
  </template>
</template>
