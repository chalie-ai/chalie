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
import { useApiClient } from '@chalie/shared';
import { withAuth } from './http';

export interface Association {
  skill_id: string | number;
  [key: string]: unknown;
}

export interface Skill {
  id: string | number;
  name: string;
  trigger?: string | null;
  instructions?: string | null;
  enabled?: boolean;
  [key: string]: unknown;
}

export interface SkillInput {
  name: string;
  trigger?: string;
  instructions?: string;
}

export const skills = {
  list(): Promise<{ skills: Skill[]; associations: Association[] }> {
    const api = useApiClient();
    return withAuth(() => api.get('/api/skills'));
  },

  create(body: SkillInput): Promise<{ skill: Skill }> {
    const api = useApiClient();
    return withAuth(() => api.post('/api/skills', body));
  },

  update(id: string | number, body: Partial<SkillInput>): Promise<{ skill: Skill }> {
    const api = useApiClient();
    return withAuth(() => api.put(`/api/skills/${id}`, body));
  },

  delete(id: string | number): Promise<unknown> {
    const api = useApiClient();
    return withAuth(() => api.del(`/api/skills/${id}`));
  },

  toggle(id: string | number): Promise<unknown> {
    const api = useApiClient();
    return withAuth(() => api.put(`/api/skills/${id}/toggle`, {}));
  },

  copy(id: string | number): Promise<{ skill: Skill }> {
    const api = useApiClient();
    return withAuth(() => api.post(`/api/skills/${id}/copy`, {}));
  },
};
