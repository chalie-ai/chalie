/**
 * Lists API — endpoints derived from frontend/brain/lists.js fetch calls.
 *
 * GET    /lists                          → { items: List[] }
 * GET    /lists/:id                      → { item: List }
 * POST   /lists                          → { item: List } create with { name }
 * DELETE /lists/:id                      → delete list
 * PUT    /lists/:id/rename               → rename list { name }
 * POST   /lists/:id/items                → add items { items: string[] }
 * PUT    /lists/:id/items/:endpoint      → toggle item (check/uncheck) { items }
 */
import { useApiClient } from '@chalie/shared';
import { withAuth } from './http';

export interface ListItem {
  id: string | number;
  content: string;
  checked: boolean;
}

export interface List {
  id: string | number;
  name: string;
  item_count?: number;
  checked_count?: number;
  items?: ListItem[];
}

export const lists = {
  list(): Promise<{ items: List[] }> {
    const api = useApiClient();
    return withAuth(() => api.get('/lists'));
  },

  get(listId: string | number): Promise<{ item: List }> {
    const api = useApiClient();
    return withAuth(() => api.get(`/lists/${listId}`));
  },

  create(name: string): Promise<{ item: List }> {
    const api = useApiClient();
    return withAuth(() => api.post('/lists', { name }));
  },

  delete(listId: string | number): Promise<unknown> {
    const api = useApiClient();
    return withAuth(() => api.del(`/lists/${listId}`));
  },

  rename(listId: string | number, name: string): Promise<unknown> {
    const api = useApiClient();
    return withAuth(() => api.put(`/lists/${listId}/rename`, { name }));
  },

  addItems(listId: string | number, items: string[]): Promise<unknown> {
    const api = useApiClient();
    return withAuth(() => api.post(`/lists/${listId}/items`, { items }));
  },

  toggleItem(listId: string | number, endpoint: string, item: { content: string }): Promise<unknown> {
    const api = useApiClient();
    return withAuth(() => api.put(`/lists/${listId}/items/${endpoint}`, { items: [item.content] }));
  },
};
