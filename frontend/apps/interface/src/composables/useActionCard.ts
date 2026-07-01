/**
 * useActionCard — singleton for ACT-cycle (chalie:action) rich-card responses.
 *
 * The card list lives OUTSIDE the conversation feed so it never pollutes
 * sortedBlocks. ActionCardHost in App.vue renders the cards once; this
 * composable is the only owner of the cards array.
 */
import { reactive } from 'vue';
import type { WsMessageEvent } from '@chalie/shared';
import { getWebSocket } from '@chalie/shared';
import { showToast } from '../utils/toast';

export interface ActionCard {
  id: number;
  status: 'acting' | 'done';
  content?: string;
}

let _nextId = 1;
let _busy = false;

const _cards = reactive<ActionCard[]>([]);

export interface ActionCardApi {
  readonly cards: ActionCard[];
  run(payload: Record<string, unknown>, onErrorMessage?: (msg: string) => void): void;
}

const _api: ActionCardApi = {
  get cards() { return _cards; },

  run(payload, onErrorMessage) {
    if (_busy) return;
    _busy = true;

    const id = _nextId++;
    _cards.push({ id, status: 'acting' });

    getWebSocket().sendAction(payload, {
      onMessage: (data) => {
        const d = data as WsMessageEvent & { content?: string };
        const content = d.content ?? '';
        const card = _cards.find((c) => c.id === id);
        if (!card) return;
        if (!content) {
          _cards.splice(_cards.indexOf(card), 1);
          return;
        }
        card.content = content;
        card.status = 'done';
      },
      onError: (data) => {
        const card = _cards.find((c) => c.id === id);
        if (card) _cards.splice(_cards.indexOf(card), 1);
        const msg = data.message ?? 'Something went wrong.';
        if (onErrorMessage) onErrorMessage(msg);
        else showToast(msg);
        _busy = false;
      },
      onDone: () => {
        _busy = false;
      },
    });
  },
};

export function useActionCard(): ActionCardApi {
  return _api;
}
