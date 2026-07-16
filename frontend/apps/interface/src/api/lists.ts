import { api } from '@chalie/shared';

/**
 * POST /api/lists/items/<list_id>:<item_id> — toggle a single list item's
 * checked state. Direct list-model endpoint; no WS round-trip, no chat-action
 * layer.
 */
export async function toggleItem(listId: string | number, itemId: string | number, checked: boolean): Promise<void> {
  await api.post(`/api/lists/items/${listId}:${itemId}`, { checked });
}
