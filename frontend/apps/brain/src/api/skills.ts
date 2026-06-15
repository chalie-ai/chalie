/**
 * Skills API — endpoints derived from frontend/brain/skills.js.
 *
 * GET    /api/skills              → { skills: Skill[], associations: Association[] }
 * POST   /api/skills              → create skill
 * PUT    /api/skills/:id          → update skill
 * DELETE /api/skills/:id          → delete skill
 * PUT    /api/skills/:id/toggle   → toggle skill enabled
 * POST   /api/skills/:id/copy     → duplicate skill
 */
import { api } from '@chalie/shared';

export interface Association {
  pattern_name: string;
  skill_title: string;
  rule: string;
  created_at?: string | null;
  [key: string]: unknown;
}

export interface Skill {
  id: string | number;
  title: string;
  use_for: string;
  content: string;
  tags?: string | null;
  version?: string | number;
  source?: string;
  enabled?: boolean;
  [key: string]: unknown;
}

export interface SkillInput {
  title: string;
  use_for: string;
  content: string;
  tags?: string;
}

export const skills = {
  list(): Promise<{ skills: Skill[]; associations: Association[] }> {
    return api.get('/api/skills');
  },

  create(body: SkillInput): Promise<{ skill: Skill }> {
    return api.post('/api/skills', body);
  },

  update(id: string | number, body: Partial<SkillInput>): Promise<{ skill: Skill }> {
    return api.put(`/api/skills/${id}`, body);
  },

  delete(id: string | number): Promise<unknown> {
    return api.del(`/api/skills/${id}`);
  },

  toggle(id: string | number): Promise<{ enabled: boolean }> {
    return api.put(`/api/skills/${id}/toggle`, {});
  },

  copy(id: string | number): Promise<{ skill: Skill }> {
    return api.post(`/api/skills/${id}/copy`, {});
  },
};
