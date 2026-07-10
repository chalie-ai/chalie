/**
 * Skills API — thin wrapper over the new Endpoint-contract routes.
 *
 * Envelope: every JSON response is { success, result, pagination?, error? }.
 * Listing responses carry `result` as an array plus `pagination`;
 * single-resource responses carry `result` as the bare DTO object.
 *
 * Routes consumed here:
 *   GET    /api/skills/all?page=N&limit=M   paginated listing (fetchAllPages).
 *   GET    /api/skills/associations         pattern → skill rules (listing).
 *   POST   /api/skills/-1                   create → 201, result DTO.
 *   POST   /api/skills/<id>                 update → result DTO.
 *   DELETE /api/skills/<id>                 → 204 empty body.
 *   POST   /api/skills/toggle/<id>          flip enabled → result DTO.
 *   POST   /api/skills/copy/<id>            duplicate a curated skill → 201.
 */
import { api } from '@chalie/shared';
import { fetchAllPages, type ListingEnvelope } from './paginate';

export interface Association {
  skill_title: string;
  pattern_name: string;
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

interface SingleEnvelope<T> {
  success: true;
  result: T;
}

export const skills = {
  async list(): Promise<{ skills: Skill[]; associations: Association[] }> {
    const [skillRows, associations] = await Promise.all([
      fetchAllPages<Skill>('/api/skills/all'),
      api.get<ListingEnvelope<Association>>('/api/skills/associations'),
    ]);
    return { skills: skillRows, associations: associations.result };
  },

  async create(body: SkillInput): Promise<Skill> {
    const res = await api.post<SingleEnvelope<Skill>>('/api/skills/-1', body);
    return res.result;
  },

  async update(id: string | number, body: Partial<SkillInput>): Promise<Skill> {
    const res = await api.post<SingleEnvelope<Skill>>(`/api/skills/${id}`, body);
    return res.result;
  },

  async delete(id: string | number): Promise<void> {
    // 204 → empty body; shared `api.del` tolerates this via .catch(() => ({})).
    await api.del(`/api/skills/${id}`);
  },

  async toggle(id: string | number): Promise<{ skill_id: number; enabled: boolean }> {
    const res = await api.post<SingleEnvelope<{ skill_id: number; enabled: boolean }>>(
      `/api/skills/toggle/${id}`,
      {},
    );
    return res.result;
  },

  async copy(id: string | number): Promise<Skill> {
    const res = await api.post<SingleEnvelope<Skill>>(`/api/skills/copy/${id}`, {});
    return res.result;
  },
};
