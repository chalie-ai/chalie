/**
 * Cognition API — endpoints derived from frontend/brain/cognition.js fetch calls.
 *
 * GET  /system/observability/records?source=&limit=&offset=&q=  → memory records
 * GET  /system/observability/tools                               → tool list
 * GET  /system/observability/world-state                         → world state
 * GET  /settings/personality                                     → personality tuple + voice (Action contract)
 * POST /settings/personality                                     → update personality (legacy PUT → POST)
 * GET  /system/observability/errors                              → recent errors
 * GET  /system/observability/token-usage?window=&type=           → token usage
 * GET  /system/observability/compaction                          → compaction summary
 */
import { api } from '@chalie/shared';

export interface MemoryRecord {
  created: string | null;
  last_accessed: string | null;
  key?: string | null;
  value: string | null;
}

export interface MemoryResponse {
  rows: MemoryRecord[];
  has_more: boolean;
  generated_at: string | null;
}

// Fields are derived from what the legacy cognition.js render code reads off each
// response (production-proven); an index signature keeps each shape forward-compatible.

export interface Tool {
  tool_name: string;
  count?: number | null;
  last_used_at?: string | null;
  [key: string]: unknown;
}

export interface WorldTelemetry {
  device?: { class?: string; platform?: string; screen_w?: number; screen_h?: number } | null;
  battery?: { level?: number; charging?: boolean } | null;
  location_name?: string | null;
  local_time?: string | null;
  timezone?: string | null;
  connection?: string | null;
  preferences?: { color_scheme?: string; [k: string]: unknown } | null;
  [k: string]: unknown;
}

export interface WorldState {
  inputs?: {
    telemetry?: WorldTelemetry | null;
    signals?: Record<string, { label?: string; [k: string]: unknown }> | null;
    bg_processes?: (string | Record<string, unknown>)[] | null;
    [k: string]: unknown;
  } | null;
  rendered?: string | null;
  [key: string]: unknown;
}

export interface Personality {
  tuple: [number, number, number, number, number];
  voice: string;
}

interface SingleEnvelope<T> {
  success: true;
  result: T;
}

export interface ErrorEntry {
  time?: string | null;
  timestamp?: string | null;
  message?: string;
  [key: string]: unknown;
}

export interface UsageSummary {
  total_tokens?: number;
  cache_hit_pct?: number | null;
  tokens_today?: number;
  most_active_model?: string;
  [key: string]: unknown;
}

export interface UsageEntry {
  bucket: string;
  type?: string;
  tokens_input?: number;
  tokens_cache_read?: number;
  tokens_output?: number;
  tokens_thinking?: number;
  [key: string]: unknown;
}

export interface UsageResponse {
  summary?: UsageSummary | null;
  entries?: UsageEntry[] | null;
  [key: string]: unknown;
}

export interface CompactionEntry {
  compacted_at?: string | null;
  compacted_up_to_id?: number | null;
  summary: string;
  [key: string]: unknown;
}

export const cognition = {
  memory(params: {
    source?: string;
    limit?: number;
    offset?: number;
    q?: string;
  }): Promise<MemoryResponse> {
    const p = new URLSearchParams();
    if (params.source) p.set('source', params.source);
    if (params.limit != null) p.set('limit', String(params.limit));
    if (params.offset != null) p.set('offset', String(params.offset));
    if (params.q) p.set('q', params.q);
    return api.get(`/api/system/observability/records?${p.toString()}`);
  },

  tools(): Promise<{ tools: Tool[] }> {
    return api.get('/api/system/observability/tools');
  },

  worldState(): Promise<WorldState> {
    return api.get('/api/system/observability/world-state');
  },

  async personality(): Promise<Personality> {
    const res = await api.get<SingleEnvelope<Personality>>('/api/settings/personality');
    return res.result;
  },

  async setPersonality(data: Partial<Personality>): Promise<Personality> {
    const res = await api.post<SingleEnvelope<Personality>>('/api/settings/personality', data);
    return res.result;
  },

  errors(): Promise<{ errors: ErrorEntry[] }> {
    return api.get('/api/system/observability/errors');
  },

  tokenUsage(window: string = 'day', type?: string): Promise<UsageResponse> {
    const p = new URLSearchParams({ window });
    if (type) p.set('type', type);
    return api.get(`/api/system/observability/token-usage?${p.toString()}`);
  },

  compaction(): Promise<{ compaction: CompactionEntry | null }> {
    return api.get('/api/system/observability/compaction');
  },
};
