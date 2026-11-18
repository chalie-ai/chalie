<!-- One server tab, live: the stream's JPEG frames drawn letterboxed on a
     canvas, and the user's mouse, wheel, touch, keys, typing and paste sent
     back to the page. Streams only while mounted, which is while the browser
     panel shows this tab. -->
<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from 'vue';
import { browser as browserApi } from '../../api';
import { useBrowserStore } from '../../stores/browser';
import {
  buttonName,
  heldButton,
  isPasteShortcut,
  keyMessage,
  letterbox,
  modifiersOf,
  nextPress,
  toPagePoint,
  wheelPixels,
  type ModifierKeys,
  type MouseButtonName,
  type PagePoint,
  type PageSize,
  type KeyInput,
  type Press,
  type StreamInput,
  type StreamMeta,
  type StreamServerMessage,
} from '../../utils/browserInput';

type MouseInput = Extract<StreamInput, { type: 'mouse' }>;
type WheelInput = Extract<StreamInput, { type: 'wheel' }>;

const props = defineProps<{ tabId: string }>();
const emit = defineEmits<{ meta: [meta: StreamMeta] }>();

const RETRY_FIRST_MS = 1000;
const RETRY_MAX_MS = 30_000;
/** A drag-resize fires a size per frame; the page is resized to the one it settles on. */
const SIZE_SETTLE_MS = 150;
/** A touch that travels further than this (view px) scrolls the page instead of tapping it. */
const TAP_SLOP_PX = 8;
/** The most characters one text message may carry; the server refuses longer ones. */
const TEXT_MAX = 4096;

const browser = useBrowserStore();
const canvasRef = ref<HTMLCanvasElement | null>(null);
const keysRef = ref<HTMLTextAreaElement | null>(null);
/** The socket is open and a frame is on the canvas. */
const live = ref(false);
/** The server closed this tab; nothing more will stream. */
const ended = ref(false);

let socket: WebSocket | null = null;
let retryDelay = RETRY_FIRST_MS;
let retryTimer: ReturnType<typeof setTimeout> | null = null;
let disposed = false;
let observer: ResizeObserver | null = null;
let sizeTimer: ReturnType<typeof setTimeout> | null = null;

let meta: StreamMeta | null = null;
let frame: ImageBitmap | null = null;
let nextFrame: Blob | null = null;
let decoding = false;

let lastPress: Press | null = null;
let pressedPointer: number | null = null;
let touch: {
  pointerId: number;
  startX: number;
  startY: number;
  lastX: number;
  lastY: number;
  scrolling: boolean;
} | null = null;
let composing = false;
/** The on-screen keyboard is wanted: the field raises it, and a tap focuses the field. */
let keyboard = false;
/** Keys sent down and not yet up, by code, so each gets its keyUp even if the field loses focus first. */
const held = new Map<string, KeyInput>();

// Moves and wheel deltas go out at most once per animation frame.
let pendingMove: MouseInput | null = null;
let pendingWheel: WheelInput | null = null;
let frameRequest = 0;

function send(msg: StreamInput): void {
  if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(msg));
}

function retryLater(): void {
  retryTimer = setTimeout(connect, retryDelay);
  retryDelay = Math.min(retryDelay * 1.5, RETRY_MAX_MS);
}

/**
 * Open the stream. The backoff resets only once the stream has sent
 * something, so a server that accepts the socket and drops it at once is
 * retried at the growing delay, not every second.
 */
function connect(): void {
  retryTimer = null;
  let ws: WebSocket;
  try {
    ws = new WebSocket(browserApi.streamUrl(props.tabId));
  } catch (err) {
    console.warn('[browser] stream socket failed to open:', err);
    retryLater();
    return;
  }
  ws.binaryType = 'blob';
  ws.onopen = sendSize;
  ws.onmessage = (e: MessageEvent<string | Blob>) => {
    if (typeof e.data === 'string') {
      onServerMessage(e.data);
      return;
    }
    retryDelay = RETRY_FIRST_MS;
    nextFrame = e.data;
    void drawFrames();
  };
  ws.onclose = (e: CloseEvent) => {
    if (socket !== ws) return;
    socket = null;
    live.value = false;
    if (disposed) return;
    // A 4xxx code is the server's final word on this stream, as `closed` is.
    if (e.code >= 4000 && e.code <= 4999) ended.value = true;
    // The tab may have closed while the socket was down; re-reading the tabs
    // moves the panel off it, which unmounts this view.
    void browser.refresh();
    if (!ended.value) retryLater();
  };
  socket = ws;
}

function onServerMessage(text: string): void {
  let msg: StreamServerMessage;
  try {
    msg = JSON.parse(text) as StreamServerMessage;
  } catch (err) {
    console.error('[browser] unreadable stream message dropped:', text, err);
    return;
  }
  switch (msg.type) {
    case 'meta':
      if (!(msg.width > 0 && msg.height > 0)) {
        console.error('[browser] stream meta without a page size dropped:', msg);
        return;
      }
      retryDelay = RETRY_FIRST_MS;
      meta = msg;
      emit('meta', msg);
      browser.patchTab(props.tabId, msg.url, msg.title);
      return;
    case 'closed':
      ended.value = true;
      socket?.close();
      return;
    case 'ping':
      return;
    default:
      console.warn('[browser] unknown stream message ignored:', msg);
  }
}

/** The page's size in CSS px: the stream's word for it, else the frame's own. */
function pageSize(): PageSize | null {
  if (meta) return meta;
  return frame ? { width: frame.width, height: frame.height } : null;
}

/**
 * Decode and draw frames one at a time. A frame that arrives mid-decode
 * replaces any still waiting, so a slow decode skips to the newest frame
 * instead of falling behind.
 */
async function drawFrames(): Promise<void> {
  if (decoding) return;
  decoding = true;
  while (nextFrame && !disposed) {
    const blob = nextFrame;
    nextFrame = null;
    try {
      const bitmap = await createImageBitmap(blob);
      if (disposed) {
        bitmap.close();
        break;
      }
      frame?.close();
      frame = bitmap;
      draw();
      live.value = socket?.readyState === WebSocket.OPEN;
    } catch (err) {
      console.warn('[browser] frame decode failed:', err);
    }
  }
  decoding = false;
}

/** Each frame fits by its own size: a resized page's new size arrives before its first frame at that size. */
function draw(): void {
  const canvas = canvasRef.value;
  const ctx = canvas?.getContext('2d');
  if (!canvas || !ctx || !frame) return;
  const box = letterbox(canvas.width, canvas.height, frame);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(frame, box.x, box.y, box.width, box.height);
}

/** Make the page's window this view: its CSS px at this screen's density. */
function sendSize(): void {
  const canvas = canvasRef.value;
  if (canvas) {
    send({
      type: 'size',
      width: canvas.clientWidth,
      height: canvas.clientHeight,
      scale: globalThis.devicePixelRatio || 1,
    });
  }
}

function queueSize(): void {
  if (sizeTimer) clearTimeout(sizeTimer);
  sizeTimer = setTimeout(() => {
    sizeTimer = null;
    sendSize();
  }, SIZE_SETTLE_MS);
}

/** Keep the canvas backing store at device px, so frames stay sharp on dense screens. */
function resize(): void {
  const canvas = canvasRef.value;
  if (!canvas) return;
  const ratio = globalThis.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(canvas.clientWidth * ratio));
  const height = Math.max(1, Math.round(canvas.clientHeight * ratio));
  if (width === canvas.width && height === canvas.height) return;
  canvas.width = width;
  canvas.height = height;
  queueSize();
  draw();
}

function pagePoint(e: { clientX: number; clientY: number }): PagePoint | null {
  const canvas = canvasRef.value;
  const page = pageSize();
  if (!canvas || !page) return null;
  const rect = canvas.getBoundingClientRect();
  return toPagePoint(e.clientX - rect.left, e.clientY - rect.top, rect.width, rect.height, page);
}

function mouse(
  event: MouseInput['event'],
  point: PagePoint,
  button: MouseButtonName,
  clickCount: number,
  mods: ModifierKeys,
): MouseInput {
  return {
    type: 'mouse',
    event,
    x: point.x,
    y: point.y,
    button,
    clickCount,
    modifiers: modifiersOf(mods),
  };
}

function flushPending(): void {
  if (frameRequest) cancelAnimationFrame(frameRequest);
  frameRequest = 0;
  if (pendingMove) send(pendingMove);
  if (pendingWheel) send(pendingWheel);
  pendingMove = null;
  pendingWheel = null;
}

function schedule(): void {
  if (!frameRequest) frameRequest = requestAnimationFrame(flushPending);
}

function queueWheel(point: PagePoint, deltaX: number, deltaY: number, modifiers: number): void {
  pendingWheel = {
    type: 'wheel',
    x: point.x,
    y: point.y,
    deltaX: deltaX + (pendingWheel?.deltaX ?? 0),
    deltaY: deltaY + (pendingWheel?.deltaY ?? 0),
    modifiers,
  };
  schedule();
}

/**
 * Keys go to the page from here on. The canvas cancels its own mousedown, a
 * tap's included, so the focus is not moved straight back off the field.
 */
function focusPage(): void {
  keysRef.value?.focus({ preventScroll: true });
}

/**
 * Raise or drop the on-screen keyboard. The field asks for none until the
 * user wants one, so a tap on the page does not raise it and shrink the
 * view. A field already focused only raises the keyboard when focused
 * again, hence the blur first.
 */
function setKeyboard(on: boolean): void {
  keyboard = on;
  const keys = keysRef.value;
  if (!keys) return;
  keys.inputMode = on ? 'text' : 'none';
  if (!on) return;
  keys.blur();
  focusPage();
}

function onPointerDown(e: PointerEvent): void {
  if (e.pointerType === 'touch') {
    if (e.isPrimary) {
      touch = {
        pointerId: e.pointerId,
        startX: e.clientX,
        startY: e.clientY,
        lastX: e.clientX,
        lastY: e.clientY,
        scrolling: false,
      };
    }
    return;
  }
  const point = pagePoint(e);
  if (!point?.inside) return;
  focusPage();
  canvasRef.value?.setPointerCapture(e.pointerId);
  pressedPointer = e.pointerId;
  lastPress = nextPress(lastPress, e.timeStamp, e.clientX, e.clientY, e.button);
  flushPending();
  send(mouse('mousePressed', point, buttonName(e.button), lastPress.count, e));
}

function onPointerMove(e: PointerEvent): void {
  const point = pagePoint(e);
  if (!point) return;
  if (e.pointerType !== 'touch') {
    pendingMove = mouse('mouseMoved', point, heldButton(e.buttons), 0, e);
    schedule();
    return;
  }
  if (touch?.pointerId !== e.pointerId) return;
  if (
    !touch.scrolling &&
    Math.hypot(e.clientX - touch.startX, e.clientY - touch.startY) < TAP_SLOP_PX
  )
    return;
  touch.scrolling = true;
  const canvas = canvasRef.value;
  const page = pageSize();
  if (!canvas || !page) return;
  // The page follows the finger, so a drag in view px is that much page, unscaled.
  const { scale } = letterbox(canvas.clientWidth, canvas.clientHeight, page);
  queueWheel(point, (touch.lastX - e.clientX) / scale, (touch.lastY - e.clientY) / scale, 0);
  touch.lastX = e.clientX;
  touch.lastY = e.clientY;
}

function onPointerUp(e: PointerEvent): void {
  if (e.pointerType === 'touch') {
    if (touch?.pointerId !== e.pointerId) return;
    const tapped = !touch.scrolling;
    touch = null;
    const point = pagePoint(e);
    if (!tapped || !point?.inside) return;
    if (keyboard) focusPage();
    flushPending();
    send(mouse('mouseMoved', point, 'none', 0, e));
    send(mouse('mousePressed', point, 'left', 1, e));
    send(mouse('mouseReleased', point, 'left', 1, e));
    return;
  }
  if (e.pointerId !== pressedPointer || !lastPress) return;
  pressedPointer = null;
  const point = pagePoint(e);
  flushPending();
  if (point) send(mouse('mouseReleased', point, buttonName(lastPress.button), lastPress.count, e));
}

function onPointerCancel(e: PointerEvent): void {
  if (e.pointerType === 'touch') touch = null;
  else onPointerUp(e);
}

function onWheel(e: WheelEvent): void {
  const page = pageSize();
  const point = pagePoint(e);
  if (!page || !point) return;
  e.preventDefault();
  queueWheel(
    point,
    wheelPixels(e.deltaX, e.deltaMode, page.height),
    wheelPixels(e.deltaY, e.deltaMode, page.height),
    modifiersOf(e),
  );
}

/**
 * Every key but F6 goes to the page, Escape and Tab included, and none
 * reaches the app's own shortcuts; F6 is left to the panel, which moves the
 * focus out to its toolbar with it. Composition keys are left to the IME,
 * whose text goes out on compositionend; the paste chord is left to the
 * browser, whose paste event carries the clipboard. A key goes up on the
 * page only if it went down there, so the paste chord's V never does.
 */
function onKey(e: KeyboardEvent, event: KeyInput['event']): void {
  if (e.key === 'F6') return;
  e.stopPropagation();
  if (e.isComposing || e.keyCode === 229) return;
  const id = e.code || e.key;
  if (event === 'keyUp') {
    if (!held.delete(id)) return;
  } else if (isPasteShortcut(e)) {
    return;
  }
  e.preventDefault();
  const msg = keyMessage(event, e);
  if (event === 'keyDown') held.set(id, msg);
  send(msg);
}

/** The field lost the keys: every key still down on the page goes up, so none stays stuck. */
function releaseKeys(): void {
  for (const down of held.values()) send({ ...down, event: 'keyUp', text: null, modifiers: 0 });
  held.clear();
}

function clearKeys(): void {
  if (keysRef.value) keysRef.value.value = '';
}

/** Text that reached the field without a forwarded key: an on-screen keyboard's. */
function onInput(): void {
  const text = keysRef.value?.value;
  if (composing || !text) return;
  sendText(text);
  clearKeys();
}

function onCompositionStart(): void {
  composing = true;
}

function onCompositionEnd(e: CompositionEvent): void {
  composing = false;
  if (e.data) sendText(e.data);
  clearKeys();
}

function onPaste(e: ClipboardEvent): void {
  e.preventDefault();
  const text = e.clipboardData?.getData('text/plain');
  if (text) sendText(text);
}

/** Types text into the page, split by code point so a long paste fits the server's limit. */
function sendText(text: string): void {
  const chars = Array.from(text);
  for (let i = 0; i < chars.length; i += TEXT_MAX) {
    send({ type: 'text', text: chars.slice(i, i + TEXT_MAX).join('') });
  }
}

onMounted(() => {
  const canvas = canvasRef.value;
  if (canvas) {
    resize();
    observer = new ResizeObserver(resize);
    observer.observe(canvas);
    canvas.addEventListener('wheel', onWheel, { passive: false });
  }
  connect();
});

onBeforeUnmount(() => {
  releaseKeys();
  disposed = true;
  if (retryTimer) clearTimeout(retryTimer);
  if (sizeTimer) clearTimeout(sizeTimer);
  if (frameRequest) cancelAnimationFrame(frameRequest);
  observer?.disconnect();
  canvasRef.value?.removeEventListener('wheel', onWheel);
  const ws = socket;
  socket = null;
  ws?.close();
  frame?.close();
  frame = null;
});

defineExpose({ focusPage, setKeyboard });
</script>

<template>
  <div class="browser-view">
    <canvas
      ref="canvasRef"
      class="browser-view__canvas"
      role="img"
      aria-label="Live page"
      @pointerdown="onPointerDown"
      @pointermove="onPointerMove"
      @pointerup="onPointerUp"
      @pointercancel="onPointerCancel"
      @mousedown.prevent
      @contextmenu.prevent
    />
    <textarea
      ref="keysRef"
      class="browser-view__keys"
      aria-label="Type into the page"
      inputmode="none"
      autocomplete="off"
      autocorrect="off"
      autocapitalize="off"
      spellcheck="false"
      @keydown="onKey($event, 'keyDown')"
      @keyup="onKey($event, 'keyUp')"
      @blur="releaseKeys"
      @input="onInput"
      @compositionstart="onCompositionStart"
      @compositionend="onCompositionEnd"
      @paste="onPaste"
    />
    <p v-if="ended" class="browser-view__note">This tab closed.</p>
    <output v-else-if="!live" class="browser-view__spinner" aria-label="Connecting to the page" />
  </div>
</template>

<style scoped lang="scss">
.browser-view {
  position: relative;
  width: 100%;
  height: 100%;
}

.browser-view__canvas {
  display: block;
  width: 100%;
  height: 100%;
  touch-action: none;
}

// Holds keyboard focus for the page: present to the keyboard, invisible on
// screen. 16px text keeps a phone from zooming in when it takes focus.
.browser-view__keys {
  position: absolute;
  top: 0;
  left: 0;
  width: 1px;
  height: 1px;
  padding: 0;
  border: 0;
  opacity: 0;
  resize: none;
  font-size: 16px;
  caret-color: transparent;
  pointer-events: none;
}

.browser-view__spinner {
  position: absolute;
  top: 50%;
  left: 50%;
  width: 20px;
  height: 20px;
  margin: -10px 0 0 -10px;
  border: 2px solid var(--line);
  border-top-color: var(--pink-text);
  border-radius: 50%;
  animation: spin 0.7s linear infinite;
}

.browser-view__note {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 0;
  color: var(--muted);
}
</style>
