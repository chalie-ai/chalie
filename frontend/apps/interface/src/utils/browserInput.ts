/**
 * The live browser view's wire format and its pure geometry: where the page
 * sits in the view, how a local mouse, wheel or key event becomes the message
 * the stream socket sends to the server's page, and how a typed address
 * becomes the url the page opens.
 */

/** The page's viewport in CSS px, as the stream's `meta` reports it. */
export interface PageSize {
  width: number;
  height: number;
}

/** Server → client text messages on the stream socket; frames arrive as binary JPEGs. */
export type StreamMeta = {
  type: 'meta';
  url: string;
  title: string;
  can_go_back: boolean;
  can_go_forward: boolean;
} & PageSize;

export type StreamServerMessage = StreamMeta | { type: 'closed' } | { type: 'ping' };

export type MouseButtonName = 'left' | 'middle' | 'right' | 'none';

/** Client → server messages. Coordinates are page CSS px; `size` is the view in device px. */
export type StreamInput =
  | {
      type: 'mouse';
      event: 'mousePressed' | 'mouseReleased' | 'mouseMoved';
      x: number;
      y: number;
      button: MouseButtonName;
      clickCount: number;
      modifiers: number;
    }
  | { type: 'wheel'; x: number; y: number; deltaX: number; deltaY: number; modifiers: number }
  | {
      type: 'key';
      event: 'keyDown' | 'keyUp';
      key: string;
      code: string;
      text: string | null;
      keyCode: number;
      modifiers: number;
    }
  | { type: 'text'; text: string }
  | ({ type: 'size' } & PageSize);

/** The page's rectangle inside a view: scaled to fit whole, centred, bars on the spare side. */
export interface Letterbox {
  x: number;
  y: number;
  width: number;
  height: number;
  /** View px per page px. */
  scale: number;
}

export function letterbox(viewWidth: number, viewHeight: number, page: PageSize): Letterbox {
  const scale = Math.min(viewWidth / page.width, viewHeight / page.height);
  const width = page.width * scale;
  const height = page.height * scale;
  return { x: (viewWidth - width) / 2, y: (viewHeight - height) / 2, width, height, scale };
}

/** A view point in page CSS px, clamped onto the page; `inside` is false when it fell on a bar. */
export interface PagePoint {
  x: number;
  y: number;
  inside: boolean;
}

/** Map a point given in the view's px (from its top-left corner) onto the page. */
export function toPagePoint(
  x: number,
  y: number,
  viewWidth: number,
  viewHeight: number,
  page: PageSize,
): PagePoint {
  const box = letterbox(viewWidth, viewHeight, page);
  const px = (x - box.x) / box.scale;
  const py = (y - box.y) / box.scale;
  return {
    x: Math.min(Math.max(px, 0), page.width),
    y: Math.min(Math.max(py, 0), page.height),
    inside: px >= 0 && py >= 0 && px <= page.width && py <= page.height,
  };
}

export interface ModifierKeys {
  altKey: boolean;
  ctrlKey: boolean;
  metaKey: boolean;
  shiftKey: boolean;
}

/** The modifier bitmask the page expects: Alt=1, Ctrl=2, Meta=4, Shift=8. */
export function modifiersOf(e: ModifierKeys): number {
  return (e.altKey ? 1 : 0) | (e.ctrlKey ? 2 : 0) | (e.metaKey ? 4 : 0) | (e.shiftKey ? 8 : 0);
}

/** A MouseEvent `button` (0 main, 1 middle, 2 secondary) by name. */
export function buttonName(button: number): MouseButtonName {
  return (['left', 'middle', 'right'] as const)[button] ?? 'none';
}

/** The button held during a move, from a MouseEvent `buttons` mask (1 main, 2 secondary, 4 middle). */
export function heldButton(buttons: number): MouseButtonName {
  if (buttons & 1) return 'left';
  if (buttons & 4) return 'middle';
  if (buttons & 2) return 'right';
  return 'none';
}

/** A mouse press, kept to tell the next one's place in a run of clicks. */
export interface Press {
  time: number;
  x: number;
  y: number;
  button: number;
  count: number;
}

const MULTI_CLICK_MS = 500;
const MULTI_CLICK_SLOP_PX = 4;

/**
 * A press of the same button soon after the last one and next to it continues
 * the run (count 2 is a double click, 3 a triple), as the page needs to select
 * a word or a line. Pointer events carry no click count of their own.
 */
export function nextPress(
  last: Press | null,
  time: number,
  x: number,
  y: number,
  button: number,
): Press {
  if (
    last &&
    last.button === button &&
    time - last.time <= MULTI_CLICK_MS &&
    Math.hypot(x - last.x, y - last.y) <= MULTI_CLICK_SLOP_PX
  ) {
    return { time, x, y, button, count: last.count + 1 };
  }
  return { time, x, y, button, count: 1 };
}

/** Wheel deltas in px: line and page deltas (some browsers' wheel mice) are converted. */
export function wheelPixels(delta: number, deltaMode: number, pageHeight: number): number {
  if (deltaMode === 1) return delta * 40;
  if (deltaMode === 2) return delta * pageHeight;
  return delta;
}

export interface KeyLike extends ModifierKeys {
  key: string;
  code: string;
  keyCode: number;
}

/**
 * The text a key types: a single character, or a carriage return for Enter
 * (which is what submits a form). A Ctrl or Meta chord is a shortcut and
 * types nothing — except Ctrl+Alt, which is AltGr on Windows.
 */
function typedText(e: KeyLike): string | null {
  if (e.metaKey || (e.ctrlKey && !e.altKey)) return null;
  if (e.key === 'Enter') return '\r';
  return [...e.key].length === 1 ? e.key : null;
}

export type KeyInput = Extract<StreamInput, { type: 'key' }>;

export function keyMessage(event: KeyInput['event'], e: KeyLike): KeyInput {
  return {
    type: 'key',
    event,
    key: e.key,
    code: e.code,
    text: event === 'keyDown' ? typedText(e) : null,
    keyCode: e.keyCode,
    modifiers: modifiersOf(e),
  };
}

/** Ctrl+V / Cmd+V: left to the browser so its paste event carries the clipboard to the page as text. */
export function isPasteShortcut(e: KeyLike): boolean {
  return (e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'v';
}

/** The url an address bar opens: typed text with no `scheme://` (and not about:, data: or file:) gets https:// in front. */
export function withScheme(typed: string): string {
  return /^([a-z][a-z\d+.-]*:\/\/|(about|data|file):)/i.test(typed) ? typed : `https://${typed}`;
}
