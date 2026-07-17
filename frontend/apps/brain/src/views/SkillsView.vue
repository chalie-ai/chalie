<script setup lang="ts">
import { computed, ref } from 'vue';
import type { Association, Skill } from '../api/skills';
import { skills as skillsApi } from '../api/skills';
import { formatDate } from '../utils/format';
import { useToast } from '../composables/useToast';
import { useBrainAction } from '../composables/useBrainAction';
import { useConfirm } from '../composables/useConfirm';
import { useBrainResource } from '../composables/useBrainResource';
import {
  BookOpen,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Copy,
  Plus,
  SquarePen,
  Trash2,
} from '@lucide/vue';

const { show: showToast } = useToast();
const { run } = useBrainAction();
const { confirm } = useConfirm();

interface SkillsPayload {
  skills: Skill[];
  associations: Association[];
}

const {
  data: skillsPayload,
  loading,
  reload: load,
} = useBrainResource(
  async () => {
    const d = await skillsApi.list();
    return { skills: d.skills ?? [], associations: d.associations ?? [] } satisfies SkillsPayload;
  },
  { initial: { skills: [], associations: [] } as SkillsPayload, failMsg: 'Failed to load skills' },
);

const skills = computed(() => skillsPayload.value.skills);
const associations = computed(() => skillsPayload.value.associations);

const expandedId = ref<string | number | null>(null);
const editingId = ref<string | number | null>(null);
const viewMode = ref<'list' | 'create'>('list');

const userSkills = computed<Skill[]>(() => skills.value.filter((s) => s.source === 'user'));
const curatedSkills = computed<Skill[]>(() => skills.value.filter((s) => s.source === 'curated'));

const createTitle = ref('');
const createUseFor = ref('');
const createTags = ref('');
const createContent = ref('');

// Only one card is ever in edit mode at a time, so these are shared.
const editUseFor = ref('');
const editTags = ref('');
const editContent = ref('');

function toggleExpand(skill: Skill): void {
  expandedId.value = expandedId.value === skill.id ? null : skill.id;
}

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
  const { ok, data: updated } = await run(
    () => skillsApi.update(skill.id, { use_for, content, tags }),
    { success: 'Skill updated', failMsg: 'Failed to update skill' },
  );
  if (ok) {
    skillsPayload.value.skills = skillsPayload.value.skills.map((s) =>
      s.id === skill.id ? updated : s,
    );
    editingId.value = null;
  }
}

async function toggleSkill(skill: Skill): Promise<void> {
  const { ok, data: t } = await run(() => skillsApi.toggle(skill.id), {
    failMsg: 'Failed to toggle skill',
  });
  if (ok) {
    skillsPayload.value.skills = skillsPayload.value.skills.map((s) =>
      s.id === skill.id ? { ...s, enabled: t.enabled } : s,
    );
    showToast(t.enabled ? 'Skill enabled' : 'Skill disabled', 'success');
  }
}

async function deleteSkill(skill: Skill): Promise<void> {
  const ok = await confirm({
    title: 'Delete Skill',
    desc: `Delete "${skill.title}"? This cannot be undone.`,
    confirmLabel: 'Delete',
    confirmClass: 'btn-danger',
  });
  if (!ok) return;
  const { ok: done } = await run(() => skillsApi.delete(skill.id), {
    success: 'Skill deleted',
    failMsg: 'Failed to delete skill',
  });
  if (done) {
    skillsPayload.value.skills = skillsPayload.value.skills.filter((s) => s.id !== skill.id);
    editingId.value = null;
  }
}

async function copySkill(skill: Skill): Promise<void> {
  const ok = await confirm({
    title: 'Customise Skill',
    desc: `Copy "${skill.title}" as a customisable skill? The curated version will be disabled.`,
    confirmLabel: 'Customise',
    confirmClass: 'btn-primary',
  });
  if (!ok) return;
  const { ok: done, data: copied } = await run(() => skillsApi.copy(skill.id), {
    failMsg: 'Failed to copy skill',
  });
  if (done) {
    showToast(`Skill copied as "${copied.title}"`, 'success');
    await load();
  }
}

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
  const { ok, data: newSkill } = await run(
    () => skillsApi.create({ title, use_for, content, tags }),
    { success: `Skill "${title}" created`, failMsg: 'Failed to create skill' },
  );
  if (ok) {
    skillsPayload.value.skills = [...skillsPayload.value.skills, newSkill];
    editingId.value = null;
    viewMode.value = 'list';
  }
}
</script>

<template>
  <div class="panel-header">
    <h2><BookOpen :size="20" /> Skills</h2>
    <div class="panel-header-actions">
      <button class="btn btn-primary btn-sm" @click="openCreate">
        <Plus :size="14" /> New Skill
      </button>
    </div>
  </div>

  <div v-if="loading" class="loading">Loading…</div>

  <template v-else-if="viewMode === 'create'">
    <div class="provider-form-page">
      <div class="form-page-header">
        <button class="btn btn-secondary btn-sm" @click="viewMode = 'list'">
          <ChevronLeft :size="14" /> Back
        </button>
        <h3>New Skill</h3>
      </div>
      <form @submit.prevent="submitCreate">
        <div class="form-group">
          <label for="skillTitle">Title <span class="text-error">*</span></label>
          <input
            id="skillTitle"
            v-model="createTitle"
            type="text"
            placeholder="e.g. Track Package Delivery"
            required
          />
        </div>
        <div class="form-group">
          <label for="skillUseFor">Use for <span class="text-error">*</span></label>
          <input
            id="skillUseFor"
            v-model="createUseFor"
            type="text"
            placeholder="One sentence: when should this skill be used?"
            required
          />
        </div>
        <div class="form-group">
          <label for="skillTags">Tags</label>
          <input
            id="skillTags"
            v-model="createTags"
            type="text"
            placeholder="logistics, tracking, delivery"
          />
        </div>
        <div class="form-group">
          <label for="skillInstructions">Instructions <span class="text-error">*</span></label>
          <textarea
            id="skillInstructions"
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

  <template v-else>
    <h4 class="section-head">My Skills</h4>

    <div v-if="userSkills.length === 0" class="empty-state mb-lg">
      <div class="empty-icon">
        <BookOpen :size="40" />
      </div>
      <h3>No custom skills yet</h3>
      <p>Click "New Skill" to create a step-by-step playbook for a recurring task.</p>
    </div>

    <div v-else id="userSkillsGrid" class="skills-grid">
      <template v-for="skill in userSkills" :key="skill.id">
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
              />
            </div>
            <div class="form-group">
              <label>Tags</label>
              <input
                v-model="editTags"
                type="text"
                class="skill-field-tags"
                placeholder="tag1, tag2"
              />
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
              <button type="button" class="btn btn-secondary btn-sm" @click="editingId = null">
                Cancel
              </button>
              <button type="submit" class="btn btn-primary btn-sm">Save</button>
            </div>
          </form>
        </div>

        <div
          v-else
          class="cap-card skill-card"
          :class="{ 'skill-expanded': expandedId === skill.id }"
        >
          <div class="skill-card-header">
            <div class="skill-card-title clickable" @click="toggleExpand(skill)">
              <span class="skill-expand-icon">
                <component :is="expandedId === skill.id ? ChevronDown : ChevronRight" :size="12" />
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
                  />
                  <span class="switch-track"></span>
                </label>
              </label>
              <button class="btn btn-secondary btn-sm" @click="startEdit(skill)">
                <SquarePen :size="13" />
              </button>
              <button class="btn btn-danger btn-sm" @click="deleteSkill(skill)">
                <Trash2 :size="13" />
              </button>
            </div>
          </div>
          <div class="skill-use-for">{{ skill.use_for }}</div>
          <div v-if="skill.tags" class="skill-tags">
            <span
              v-for="tag in skill.tags
                .split(',')
                .map((t) => t.trim())
                .filter(Boolean)"
              :key="tag"
              class="badge badge-muted skill-tag"
              >{{ tag }}</span
            >
          </div>
          <div v-if="expandedId === skill.id" class="skill-expanded-content">
            <pre class="skill-content-preview">{{ skill.content }}</pre>
          </div>
        </div>
      </template>
    </div>

    <h4 class="section-head mt-lg">Curated Skills</h4>

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
          <div class="skill-card-title clickable" @click="toggleExpand(skill)">
            <span class="skill-expand-icon">
              <component :is="expandedId === skill.id ? ChevronDown : ChevronRight" :size="12" />
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
                />
                <span class="switch-track"></span>
              </label>
            </label>
            <button
              class="btn btn-secondary btn-sm"
              title="Copy &amp; Customise"
              @click="copySkill(skill)"
            >
              <Copy :size="13" /> Customise
            </button>
          </div>
        </div>
        <div class="skill-use-for">{{ skill.use_for }}</div>
        <div v-if="skill.tags" class="skill-tags">
          <span
            v-for="tag in skill.tags
              .split(',')
              .map((t) => t.trim())
              .filter(Boolean)"
            :key="tag"
            class="badge badge-muted skill-tag"
            >{{ tag }}</span
          >
        </div>
        <div v-if="expandedId === skill.id" class="skill-expanded-content">
          <pre class="skill-content-preview">{{ skill.content }}</pre>
        </div>
      </div>
    </div>

    <template v-if="associations.length > 0">
      <h4 class="section-head mt-lg">Skill Associations</h4>
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
            <td>
              <span class="badge badge-violet">{{ a.pattern_name }}</span>
            </td>
            <td>{{ a.skill_title }}</td>
            <td class="text-xs text-secondary">{{ a.rule }}</td>
            <td class="text-xs text-tertiary">{{ formatDate(a.created_at) }}</td>
          </tr>
        </tbody>
      </table>
    </template>
  </template>
</template>
