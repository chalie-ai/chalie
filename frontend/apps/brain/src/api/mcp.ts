/**
 * MCP API — endpoints derived from frontend/brain/mcp.js.
 *
 * Inbound (server side — Chalie as MCP server) — Endpoint-contract routes in
 * backend/api/endpoints/mcp_settings.py:
 *   GET  /api/mcp-server/all                      one-item listing → settings + live listener state.
 *   POST /api/mcp-server/-1                       partial update → 204 empty body, applied live.
 *
 * Outbound (client side — Chalie connecting to external MCP servers) — thin
 * wrapper over the Endpoint-contract routes in
 * backend/api/endpoints/mcp_clients.py + backend/api/actions/mcp_clients/*.
 * Envelope: every JSON response is { success, result, pagination?, error? }.
 *
 *   GET    /api/mcp-clients/all?page=N&limit=M   paginated listing (fetchAllPages).
 *   POST   /api/mcp-clients/-1                   add client server → 201, result DTO.
 *   POST   /api/mcp-clients/<id>                 update client server → result DTO.
 *   DELETE /api/mcp-clients/<id>                 → 204 empty body.
 *   POST   /api/mcp-clients/test/<id>            test connection → result DTO.
 */
import { api } from '@chalie/shared';
import { fetchAllPages, type ListingEnvelope } from './paginate';

export interface McpServerConfig {
  enabled?: boolean;
  port?: number;
  listening?: boolean;
  listening_port?: number | null;
  error?: string | null;
  [key: string]: unknown;
}

export interface McpClient {
  id: string | number;
  name: string;
  host: string;
  status?: string;
  enabled?: boolean;
  headers?: Record<string, string>;
  [key: string]: unknown;
}

export interface McpClientInput {
  name: string;
  host: string;
  headers: Record<string, string>;
  enabled?: boolean;
  [key: string]: unknown;
}

interface SingleEnvelope<T> {
  success: true;
  result: T;
}

export const mcp = {
  async getServerConfig(): Promise<McpServerConfig> {
    const res = await api.get<ListingEnvelope<McpServerConfig>>('/api/mcp-server/all');
    return res.result[0];
  },

  updateServerConfig(body: Partial<McpServerConfig>): Promise<unknown> {
    return api.post('/api/mcp-server/-1', body);
  },

  listClients(): Promise<McpClient[]> {
    return fetchAllPages<McpClient>('/api/mcp-clients/all');
  },

  async createClient(body: McpClientInput): Promise<McpClient> {
    const res = await api.post<SingleEnvelope<McpClient>>('/api/mcp-clients/-1', body);
    return res.result;
  },

  async updateClient(id: string | number, body: Partial<McpClientInput>): Promise<McpClient> {
    const res = await api.post<SingleEnvelope<McpClient>>(`/api/mcp-clients/${id}`, body);
    return res.result;
  },

  deleteClient(id: string | number): Promise<unknown> {
    return api.del(`/api/mcp-clients/${id}`);
  },

  async testClient(id: string | number): Promise<{ reachable: boolean; tool_count?: number; status?: string }> {
    const res = await api.post<SingleEnvelope<{ reachable: boolean; tool_count?: number; status?: string }>>(
      `/api/mcp-clients/test/${id}`,
      {},
    );
    return res.result;
  },
};
