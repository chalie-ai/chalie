/**
 * MCP API — endpoints derived from frontend/brain/mcp.js.
 *
 * Inbound (server side — Chalie as MCP server):
 *   GET  /api/mcp-server                          → config
 *   PUT  /api/mcp-server                          → update config
 *   POST /api/mcp-server/regenerate-token         → regenerate token
 *
 * Outbound (client side — Chalie connecting to external MCP servers):
 *   GET    /api/mcp-clients/       → list client servers (bare array)
 *   POST   /api/mcp-clients/       → add client server
 *   PUT    /api/mcp-clients/:id    → update client server
 *   DELETE /api/mcp-clients/:id    → delete client server
 *   POST   /api/mcp-clients/:id/test → test connection
 */
import { api } from '@chalie/shared';

export interface McpServerConfig {
  enabled?: boolean;
  port?: number;
  token?: string | null;
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

export const mcp = {
  getServerConfig(): Promise<McpServerConfig> {
    return api.get('/api/mcp-server');
  },

  updateServerConfig(body: Partial<McpServerConfig>): Promise<unknown> {
    return api.put('/api/mcp-server', body);
  },

  regenerateToken(): Promise<{ token: string }> {
    return api.post('/api/mcp-server/regenerate-token', {});
  },

  listClients(): Promise<McpClient[]> {
    return api.get('/api/mcp-clients/');
  },

  createClient(body: McpClientInput): Promise<unknown> {
    return api.post('/api/mcp-clients/', body);
  },

  updateClient(id: string | number, body: Partial<McpClientInput>): Promise<unknown> {
    return api.put(`/api/mcp-clients/${id}`, body);
  },

  deleteClient(id: string | number): Promise<unknown> {
    return api.del(`/api/mcp-clients/${id}`);
  },

  testClient(id: string | number): Promise<{ reachable: boolean; tool_count?: number; status?: string }> {
    return api.post(`/api/mcp-clients/${id}/test`, {});
  },
};
