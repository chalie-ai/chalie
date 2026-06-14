/**
 * MCP API — endpoints derived from frontend/brain/mcp.js.
 *
 * Inbound (server side — Chalie as MCP server):
 *   GET  /api/mcp-server                          → config
 *   PUT  /api/mcp-server                          → update config
 *   POST /api/mcp-server/regenerate-token         → regenerate token
 *
 * Outbound (client side — Chalie connecting to external MCP servers):
 *   GET    /api/mcp-clients/       → list client servers
 *   POST   /api/mcp-clients/       → add client server
 *   PUT    /api/mcp-clients/:id    → update client server
 *   DELETE /api/mcp-clients/:id    → delete client server
 *   POST   /api/mcp-clients/:id/test → test connection
 */
import { useApiClient } from '@chalie/shared';
import { withAuth } from './http';

export interface McpServerConfig {
  enabled?: boolean;
  port?: number;
  token?: string | null;
  [key: string]: unknown;
}

export interface McpClient {
  id: string | number;
  name: string;
  url: string;
  enabled?: boolean;
  [key: string]: unknown;
}

export interface McpClientInput {
  name: string;
  url: string;
  enabled?: boolean;
  [key: string]: unknown;
}

export const mcp = {
  // ── Inbound ──────────────────────────────────────────────────────────────

  getServerConfig(): Promise<McpServerConfig> {
    const api = useApiClient();
    return withAuth(() => api.get('/api/mcp-server'));
  },

  updateServerConfig(body: Partial<McpServerConfig>): Promise<McpServerConfig> {
    const api = useApiClient();
    return withAuth(() => api.put('/api/mcp-server', body));
  },

  regenerateToken(): Promise<{ token: string }> {
    const api = useApiClient();
    return withAuth(() => api.post('/api/mcp-server/regenerate-token', {}));
  },

  // ── Outbound ─────────────────────────────────────────────────────────────

  listClients(): Promise<{ servers?: McpClient[]; clients?: McpClient[] }> {
    const api = useApiClient();
    return withAuth(() => api.get('/api/mcp-clients/'));
  },

  createClient(body: McpClientInput): Promise<{ server?: McpClient; client?: McpClient }> {
    const api = useApiClient();
    return withAuth(() => api.post('/api/mcp-clients/', body));
  },

  updateClient(id: string | number, body: Partial<McpClientInput>): Promise<unknown> {
    const api = useApiClient();
    return withAuth(() => api.put(`/api/mcp-clients/${id}`, body));
  },

  deleteClient(id: string | number): Promise<unknown> {
    const api = useApiClient();
    return withAuth(() => api.del(`/api/mcp-clients/${id}`));
  },

  testClient(id: string | number): Promise<{ success: boolean; error?: string }> {
    const api = useApiClient();
    return withAuth(() => api.post(`/api/mcp-clients/${id}/test`, {}));
  },
};
