/**
 * Lists API — thin wrapper over the new Endpoint-contract routes.
 *
 * Envelope: every JSON response is { success, result, pagination?, error? }.
 * Listing responses carry `result` as an array plus `pagination`;
 * single-resource responses carry `result` as the bare DTO object.
 * Errors use { success: false, result: [], error: "..." }.
 *
 * Routes consumed here:
 *   GET    /api/lists/all?page=N&limit=M    paginated listing (limit 1..100;
 *                                          we fetch limit=100 and loop pages
 *                                          until pagination.total collected).
 *   POST   /api/lists/-1                    create { name } → 201, result DTO.
 *   POST   /api/lists/<id>                  update (rename via { name }).
 *   DELETE /api/lists/<id>                  → 204 empty body.
 *   GET    /api/lists/items/<list_id>       list a list's items.
 *   POST   /api/lists/items/<list_id>       add ONE item { content }.
 *   POST   /api/lists/items/<list_id>:<item_id>
 *                                          update item (toggle checked).
 *
 * Deliberately keeps the exported names stable so calling views need minimal
 * changes.
 */
import { api } from '@chalie/shared';
import { fetchAllPages, type ListingEnvelope } from './paginate';

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

interface SingleEnvelope<T> {
  success: true;
  result: T;
}

export const lists = {
  list(): Promise<List[]> {
    return fetchAllPages<List>('/api/lists/all');
  },

  async items(listId: string | number): Promise<ListItem[]> {
    const body = await api.get<ListingEnvelope<ListItem>>(`/api/lists/items/${listId}`);
    return body.result;
  },

  async create(name: string): Promise<List> {
    const body = await api.post<SingleEnvelope<List>>(`/api/lists/-1`, { name });
    return body.result;
  },

  async rename(listId: string | number, name: string): Promise<List> {
    const body = await api.post<SingleEnvelope<List>>(`/api/lists/${listId}`, { name });
    return body.result;
  },

  async delete(listId: string | number): Promise<void> {
    // 204 → empty body; shared `api.del` tolerates this via .catch(() => ({})).
    await api.del(`/api/lists/${listId}`);
  },

  async addItem(listId: string | number, content: string): Promise<void> {
    await api.post(`/api/lists/items/${listId}`, { content });
  },

  async toggleItem(listId: string | number, itemId: string | number, checked: boolean): Promise<ListItem> {
    const body = await api.post<SingleEnvelope<ListItem>>(
      `/api/lists/items/${listId}:${itemId}`,
      { checked },
    );
    return body.result;
  },
};
