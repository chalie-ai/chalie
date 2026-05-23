// Skills panel — curated and user-created skill playbook management.
const PanelSkills = (() => {
  let _root = null;
  let _skills = [];
  let _associations = [];
  let _loaded = false;
  let _editingId = null;

  function mount(root) {
    _root = root;
    _render();
  }

  function unmount() {
    _root = null;
    _editingId = null;
  }

  // ── Data loading ──────────────────────────────────────────────────

  async function _load() {
    try {
      const res = await BrainApp.apiFetch('/api/skills');
      if (!res.ok) throw new Error('Failed to load skills');
      const data = await res.json();
      _skills = data.skills || [];
      _associations = data.associations || [];
      _loaded = true;
    } catch {
      BrainApp.showToast('Failed to load skills', 'error');
    }
    _renderContent();
  }

  // ── Rendering ─────────────────────────────────────────────────────

  function _render() {
    if (!_root) return;
    _root.innerHTML = `
      <div class="panel-header">
        <h2>${Icons.Skill(20)} Skills</h2>
        <div class="panel-header-actions">
          <button class="btn btn-primary btn-sm" id="skillsAddBtn">${Icons.Plus(14)} New Skill</button>
        </div>
      </div>
      <div id="skillsContent"><div class="loading">Loading…</div></div>`;

    document.getElementById('skillsAddBtn').addEventListener('click', _openCreateForm);

    if (!_loaded) _load();
    else _renderContent();
  }

  function _renderContent() {
    const el = document.getElementById('skillsContent');
    if (!el) return;

    const curated = _skills.filter(s => s.source === 'curated');
    const user = _skills.filter(s => s.source === 'user');

    let html = '';

    // ── User Skills section ──
    html += `<h4 class="section-head">My Skills</h4>`;
    if (user.length === 0) {
      html += `<div class="empty-state" style="margin-bottom:24px;">
        <div class="empty-icon">${Icons.Skill(40)}</div>
        <h3>No custom skills yet</h3>
        <p>Click "New Skill" to create a step-by-step playbook for a recurring task.</p>
      </div>`;
    } else {
      html += `<div class="skills-grid" id="userSkillsGrid">`;
      for (const skill of user) {
        html += _renderUserCard(skill);
      }
      html += `</div>`;
    }

    // ── Curated Skills section ──
    html += `<h4 class="section-head" style="margin-top:32px;">Curated Skills</h4>`;
    if (curated.length === 0) {
      html += `<div class="empty-state"><p>No curated skills loaded.</p></div>`;
    } else {
      html += `<div class="skills-grid" id="curatedSkillsGrid">`;
      for (const skill of curated) {
        html += _renderCuratedCard(skill);
      }
      html += `</div>`;
    }

    // ── Associations section ──
    if (_associations.length > 0) {
      html += `<h4 class="section-head" style="margin-top:32px;">Skill Associations</h4>
        <p class="panel-desc">Patterns discovered from your behaviour, linked to skills.</p>
        <table class="records-table">
          <thead><tr>
            <th>Pattern</th>
            <th>Skill</th>
            <th>Rule</th>
            <th>Since</th>
          </tr></thead>
          <tbody>`;
      for (const a of _associations) {
        html += `<tr>
          <td><span class="badge badge-violet">${BrainApp.escapeHtml(a.pattern_name)}</span></td>
          <td>${BrainApp.escapeHtml(a.skill_title)}</td>
          <td style="font-size:12px;color:var(--text-secondary);">${BrainApp.escapeHtml(a.rule)}</td>
          <td style="font-size:12px;color:var(--text-tertiary);">${BrainApp.formatDate(a.created_at)}</td>
        </tr>`;
      }
      html += `</tbody></table>`;
    }

    el.innerHTML = html;
    _bindEvents(el);
  }

  function _renderUserCard(skill) {
    const isEditing = _editingId === skill.id;
    const enabledBadge = skill.enabled
      ? `<span class="badge badge-success">enabled</span>`
      : `<span class="badge badge-muted">disabled</span>`;

    if (isEditing) {
      return `<div class="cap-card skill-card" data-skill-id="${skill.id}">
        <div class="skill-card-header">
          <strong>${BrainApp.escapeHtml(skill.title)}</strong>
          <span class="badge badge-violet">v${skill.version}</span>
        </div>
        <form class="skill-edit-form" data-skill-id="${skill.id}">
          <div class="form-group">
            <label>Use for</label>
            <input type="text" class="skill-field-use_for" value="${BrainApp.escapeHtml(skill.use_for)}" placeholder="When should this skill be used?">
          </div>
          <div class="form-group">
            <label>Tags</label>
            <input type="text" class="skill-field-tags" value="${BrainApp.escapeHtml(skill.tags || '')}" placeholder="tag1, tag2">
          </div>
          <div class="form-group">
            <label>Related abilities</label>
            <input type="text" class="skill-field-related_abilities" value="${BrainApp.escapeHtml(skill.related_abilities || '')}" placeholder="search, memory, schedule">
          </div>
          <div class="form-group">
            <label>Instructions</label>
            <textarea class="skill-field-content" rows="8" placeholder="1. First step&#10;2. Second step">${BrainApp.escapeHtml(skill.content)}</textarea>
          </div>
          <div class="form-actions">
            <button type="button" class="btn btn-secondary btn-sm" data-cancel-edit="${skill.id}">Cancel</button>
            <button type="submit" class="btn btn-primary btn-sm">Save</button>
          </div>
        </form>
      </div>`;
    }

    return `<div class="cap-card skill-card" data-skill-id="${skill.id}">
      <div class="skill-card-header">
        <div class="skill-card-title">
          <strong>${BrainApp.escapeHtml(skill.title)}</strong>
          <span class="badge badge-violet">v${skill.version}</span>
          ${enabledBadge}
        </div>
        <div class="skill-card-actions">
          <label class="switch-label skill-toggle-wrap" title="${skill.enabled ? 'Disable' : 'Enable'}">
            <label class="switch">
              <input type="checkbox" class="skill-toggle" data-skill-id="${skill.id}" ${skill.enabled ? 'checked' : ''}>
              <span class="switch-track"></span>
            </label>
          </label>
          <button class="btn btn-secondary btn-sm" data-edit-skill="${skill.id}">${Icons.Edit(13)}</button>
          <button class="btn btn-danger btn-sm" data-delete-skill="${skill.id}" data-skill-title="${BrainApp.escapeHtml(skill.title)}">${Icons.Trash(13)}</button>
        </div>
      </div>
      <div class="skill-use-for">${BrainApp.escapeHtml(skill.use_for)}</div>
      ${skill.tags ? `<div class="skill-tags">${_renderTags(skill.tags)}</div>` : ''}
    </div>`;
  }

  function _renderCuratedCard(skill) {
    const enabledBadge = skill.enabled
      ? `<span class="badge badge-success">enabled</span>`
      : `<span class="badge badge-muted">disabled</span>`;

    return `<div class="cap-card skill-card${!skill.enabled ? ' skill-disabled' : ''}" data-skill-id="${skill.id}">
      <div class="skill-card-header">
        <div class="skill-card-title">
          <strong>${BrainApp.escapeHtml(skill.title)}</strong>
          <span class="badge badge-muted">v${skill.version}</span>
          ${enabledBadge}
        </div>
        <div class="skill-card-actions">
          <label class="switch-label skill-toggle-wrap" title="${skill.enabled ? 'Disable' : 'Enable'}">
            <label class="switch">
              <input type="checkbox" class="skill-toggle" data-skill-id="${skill.id}" ${skill.enabled ? 'checked' : ''}>
              <span class="switch-track"></span>
            </label>
          </label>
          <button class="btn btn-secondary btn-sm" data-copy-skill="${skill.id}" data-skill-title="${BrainApp.escapeHtml(skill.title)}" title="Copy &amp; Customise">${Icons.Copy(13)} Customise</button>
        </div>
      </div>
      <div class="skill-use-for">${BrainApp.escapeHtml(skill.use_for)}</div>
      ${skill.tags ? `<div class="skill-tags">${_renderTags(skill.tags)}</div>` : ''}
    </div>`;
  }

  function _renderTags(tagsStr) {
    return tagsStr.split(',').map(t => t.trim()).filter(Boolean)
      .map(t => `<span class="badge badge-muted skill-tag">${BrainApp.escapeHtml(t)}</span>`)
      .join('');
  }

  // ── Event binding ─────────────────────────────────────────────────

  function _bindEvents(el) {
    el.querySelectorAll('.skill-toggle').forEach(cb => {
      cb.addEventListener('change', () => _toggleSkill(Number(cb.dataset.skillId)));
    });
    el.querySelectorAll('[data-edit-skill]').forEach(btn => {
      btn.addEventListener('click', () => {
        _editingId = Number(btn.dataset.editSkill);
        _renderContent();
      });
    });
    el.querySelectorAll('[data-cancel-edit]').forEach(btn => {
      btn.addEventListener('click', () => {
        _editingId = null;
        _renderContent();
      });
    });
    el.querySelectorAll('[data-delete-skill]').forEach(btn => {
      btn.addEventListener('click', () => {
        const title = btn.dataset.skillTitle;
        if (confirm(`Delete skill "${title}"? This cannot be undone.`)) {
          _deleteSkill(Number(btn.dataset.deleteSkill));
        }
      });
    });
    el.querySelectorAll('[data-copy-skill]').forEach(btn => {
      btn.addEventListener('click', () => {
        const title = btn.dataset.skillTitle;
        if (confirm(`Copy "${title}" as a customisable user skill? The curated version will be disabled.`)) {
          _copySkill(Number(btn.dataset.copySkill));
        }
      });
    });
    el.querySelectorAll('.skill-edit-form').forEach(form => {
      form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const skillId = Number(form.dataset.skillId);
        await _saveEdit(skillId, form);
      });
    });
  }

  // ── Create form ───────────────────────────────────────────────────

  function _openCreateForm() {
    const el = document.getElementById('skillsContent');
    if (!el) return;

    el.innerHTML = `
      <div class="provider-form-page">
        <div class="form-page-header">
          <button class="btn btn-secondary btn-sm" id="backToSkills">${Icons.Chevron(14)} Back</button>
          <h3>New Skill</h3>
        </div>
        <form id="createSkillForm">
          <div class="form-group">
            <label for="skill_title">Title <span style="color:var(--error)">*</span></label>
            <input type="text" id="skill_title" placeholder="e.g. Track Package Delivery" required>
          </div>
          <div class="form-group">
            <label for="skill_use_for">Use for <span style="color:var(--error)">*</span></label>
            <input type="text" id="skill_use_for" placeholder="One sentence: when should this skill be used?" required>
          </div>
          <div class="form-group">
            <label for="skill_tags">Tags</label>
            <input type="text" id="skill_tags" placeholder="logistics, tracking, delivery">
          </div>
          <div class="form-group">
            <label for="skill_related_abilities">Related abilities</label>
            <input type="text" id="skill_related_abilities" placeholder="search, memory, schedule">
          </div>
          <div class="form-group">
            <label for="skill_content">Instructions <span style="color:var(--error)">*</span></label>
            <textarea id="skill_content" rows="10" placeholder="1. First step (reference tools like \`search\`, \`memory\`)&#10;2. Second step&#10;3. Third step" required></textarea>
          </div>
          <div class="form-actions">
            <button type="button" class="btn btn-secondary" id="cancelCreateSkill">Cancel</button>
            <button type="submit" class="btn btn-primary">Create Skill</button>
          </div>
        </form>
      </div>`;

    document.getElementById('backToSkills').addEventListener('click', _renderContent);
    document.getElementById('cancelCreateSkill').addEventListener('click', _renderContent);
    document.getElementById('createSkillForm').addEventListener('submit', _submitCreate);
  }

  // ── API actions ───────────────────────────────────────────────────

  async function _submitCreate(e) {
    e.preventDefault();
    const title = document.getElementById('skill_title').value.trim();
    const use_for = document.getElementById('skill_use_for').value.trim();
    const content = document.getElementById('skill_content').value.trim();
    const tags = document.getElementById('skill_tags').value.trim();
    const related_abilities = document.getElementById('skill_related_abilities').value.trim();

    try {
      const res = await BrainApp.apiFetch('/api/skills', {
        method: 'POST',
        body: JSON.stringify({ title, use_for, content, tags, related_abilities }),
      });
      const data = await res.json();
      if (res.ok) {
        BrainApp.showToast(`Skill "${title}" created`, 'success');
        _skills.push(data.skill);
        _editingId = null;
        _renderContent();
      } else {
        BrainApp.showToast(data.error || 'Failed to create skill', 'error');
      }
    } catch {
      BrainApp.showToast('Network error', 'error');
    }
  }

  async function _saveEdit(skillId, form) {
    const use_for = form.querySelector('.skill-field-use_for').value.trim();
    const content = form.querySelector('.skill-field-content').value.trim();
    const tags = form.querySelector('.skill-field-tags').value.trim();
    const related_abilities = form.querySelector('.skill-field-related_abilities').value.trim();

    try {
      const res = await BrainApp.apiFetch(`/api/skills/${skillId}`, {
        method: 'PUT',
        body: JSON.stringify({ use_for, content, tags, related_abilities }),
      });
      const data = await res.json();
      if (res.ok) {
        BrainApp.showToast('Skill updated', 'success');
        const idx = _skills.findIndex(s => s.id === skillId);
        if (idx !== -1) _skills[idx] = data.skill;
        _editingId = null;
        _renderContent();
      } else {
        BrainApp.showToast(data.error || 'Failed to update skill', 'error');
      }
    } catch {
      BrainApp.showToast('Network error', 'error');
    }
  }

  async function _toggleSkill(skillId) {
    try {
      const res = await BrainApp.apiFetch(`/api/skills/${skillId}/toggle`, { method: 'PUT' });
      const data = await res.json();
      if (res.ok) {
        const idx = _skills.findIndex(s => s.id === skillId);
        if (idx !== -1) _skills[idx].enabled = data.enabled;
        BrainApp.showToast(data.enabled ? 'Skill enabled' : 'Skill disabled', 'success');
        _renderContent();
      } else {
        BrainApp.showToast(data.error || 'Failed to toggle skill', 'error');
      }
    } catch {
      BrainApp.showToast('Network error', 'error');
    }
  }

  async function _deleteSkill(skillId) {
    try {
      const res = await BrainApp.apiFetch(`/api/skills/${skillId}`, { method: 'DELETE' });
      const data = await res.json();
      if (res.ok) {
        BrainApp.showToast('Skill deleted', 'success');
        _skills = _skills.filter(s => s.id !== skillId);
        _editingId = null;
        _renderContent();
      } else {
        BrainApp.showToast(data.error || 'Failed to delete skill', 'error');
      }
    } catch {
      BrainApp.showToast('Network error', 'error');
    }
  }

  async function _copySkill(skillId) {
    try {
      const res = await BrainApp.apiFetch(`/api/skills/${skillId}/copy`, { method: 'POST' });
      const data = await res.json();
      if (res.ok) {
        BrainApp.showToast(`Skill copied as "${data.skill.title}"`, 'success');
        _loaded = false;
        _load();
      } else {
        BrainApp.showToast(data.error || 'Failed to copy skill', 'error');
      }
    } catch {
      BrainApp.showToast('Network error', 'error');
    }
  }

  return { mount, unmount };
})();
