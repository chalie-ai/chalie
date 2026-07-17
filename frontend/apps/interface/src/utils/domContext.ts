import { ConfigType } from '@chalie/shared';

/**
 * DOM contract reader — shared by every handler that must act on a specific
 * turn (send / stop / undo). A handler receives an HTMLElement (typically a
 * template ref on the control's root) and walks up from it with `closest()` to
 * find the nearest ancestor carrying the matching data attribute. Each key
 * resolves independently, so the three markers may live on different ancestors.
 *
 * Parsing rules:
 *   - `data-turn-id`   → Number(value), or null when absent or not a valid number
 *   - `data-type`      → value, or ConfigType.USER when absent
 *   - `data-transcript-row-id` → Number(value), or null when absent or not a valid number
 *   - `data-dock-scope` → Number(value), or null when absent or not a valid number
 *
 * Note: Vue renders `null` props as the string "null" in data attributes. We
 * treat that as absent (returning null) since it represents "no turn selected".
 */
export function readDomContext(el: HTMLElement | null): {
  turnId: number | null;
  type: string;
  transcriptRowId: number | null;
  dockScope: number | null;
} {
  if (!el) {
    return { turnId: null, type: ConfigType.USER, transcriptRowId: null, dockScope: null };
  }

  const turnIdEl = el.closest<HTMLElement>('[data-turn-id]');
  const typeEl = el.closest<HTMLElement>('[data-type]');
  const transcriptRowEl = el.closest<HTMLElement>('[data-transcript-row-id]');
  const dockScopeEl = el.closest<HTMLElement>('[data-dock-scope]');

  const parseTurnId = (attr: string | undefined): number | null => {
    if (!attr) return null;
    const num = Number(attr);
    return Number.isNaN(num) ? null : num;
  };

  const turnId = parseTurnId(turnIdEl?.dataset.turnId);
  const type = typeEl?.dataset.type ?? ConfigType.USER;
  const transcriptRowId = parseTurnId(transcriptRowEl?.dataset.transcriptRowId);
  const dockScope = parseTurnId(dockScopeEl?.dataset.dockScope);

  return { turnId, type, transcriptRowId, dockScope };
}
