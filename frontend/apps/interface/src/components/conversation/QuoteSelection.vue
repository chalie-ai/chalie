<script setup lang="ts">
/**
 * QuoteSelection — quoting a piece of Chalie's reply. Text selected within one
 * reply, on a surface with a composer, floats a Quote button over it; Quote
 * opens a one-line reply field in its place, and Enter hands the quote and the
 * reply to that surface's composer as a quote block. Escape or a click outside
 * drops it. Mounted once: each surface names its composer's dock in a
 * `data-quote-dock` attribute around its replies.
 */
import { computed, nextTick, onBeforeUnmount, onMounted, ref } from 'vue';
import { Quote } from '@lucide/vue';
import { QUOTE_BLOCK_EVENT, type QuoteBlockEventDetail } from '../../utils/quoteBlocks';

const GAP = 8;
// How long a selection must hold still, when no pointer drags it (the keyboard,
// a touch screen's handles), before the Quote button follows it.
const SETTLE_MS = 150;

const mode = ref<'idle' | 'bar' | 'reply'>('idle');
const quote = ref('');
const reply = ref('');
const pos = ref<{ top: number; left: number } | null>(null);
const barRef = ref<HTMLElement | null>(null);
const popRef = ref<HTMLElement | null>(null);
const inputRef = ref<HTMLInputElement | null>(null);

let range: Range | null = null;
let dockId = '';
let pressed = false;
let timer: ReturnType<typeof setTimeout> | undefined;

const style = computed(() =>
  pos.value
    ? { top: `${pos.value.top}px`, left: `${pos.value.left}px` }
    : { visibility: 'hidden' as const },
);

function elementOf(node: Node): Element | null {
  return node instanceof Element ? node : node.parentElement;
}

/** The selection, when it starts in the text of one of Chalie's replies on a
 *  surface with a composer and stays in that reply: its text, its range and
 *  that composer's dock. */
function readSelection(): { text: string; range: Range; dockId: string } | null {
  const sel = document.getSelection();
  if (!sel || sel.isCollapsed || !sel.rangeCount) return null;
  const r = sel.getRangeAt(0);
  const start = elementOf(r.startContainer);
  const bubble = start?.closest('.speech-form--chalie');
  if (!bubble || !start?.closest('.speech-form__text')) return null;
  const dock = bubble.closest<HTMLElement>('[data-quote-dock]')?.dataset.quoteDock;
  if (!dock) return null;
  if (!bubble.contains(r.endContainer)) {
    // A triple-click carries the end past the reply; nothing but whitespace may
    // lie beyond it, and the selection is cut back to the reply's end.
    const beyond = document.createRange();
    beyond.setStart(bubble, bubble.childNodes.length);
    beyond.setEnd(r.endContainer, r.endOffset);
    if (beyond.toString().trim()) return null;
    const cut = r.cloneRange();
    cut.setEnd(bubble, bubble.childNodes.length);
    sel.removeAllRanges();
    sel.addRange(cut);
  }
  const text = sel.toString().trim();
  return text ? { text, range: sel.getRangeAt(0).cloneRange(), dockId: dock } : null;
}

function reset(): void {
  clearTimeout(timer);
  mode.value = 'idle';
  pos.value = null;
  range = null;
}

/** Above the selection, or under it where there is no room; kept on screen. */
function place(): void {
  const el = mode.value === 'reply' ? popRef.value : barRef.value;
  const rect = range?.getBoundingClientRect();
  // A reply re-rendered under the selection leaves an empty range: stay put.
  if (!el || !rect || (!rect.width && !rect.height)) return;
  const { offsetWidth: w, offsetHeight: h } = el;
  const above = rect.top - GAP - h;
  const top = above >= GAP ? above : Math.min(rect.bottom + GAP, innerHeight - h - GAP);
  const left = Math.min(Math.max(rect.left + rect.width / 2 - w / 2, GAP), innerWidth - w - GAP);
  pos.value = { top, left };
}

function evaluate(): void {
  if (mode.value === 'reply') return;
  const found = readSelection();
  if (!found) return reset();
  ({ range, dockId } = found);
  quote.value = found.text;
  mode.value = 'bar';
  void nextTick(place);
}

function schedule(ms: number): void {
  clearTimeout(timer);
  timer = setTimeout(evaluate, ms);
}

/** The field takes the focus once it is placed: a browser will not focus it
 *  while it is still hidden, waiting to be measured. */
async function openReply(): Promise<void> {
  reply.value = '';
  mode.value = 'reply';
  pos.value = null;
  await nextTick();
  place();
  await nextTick();
  inputRef.value?.focus();
}

function commit(): void {
  const detail: QuoteBlockEventDetail = {
    dockId,
    block: { quote: quote.value, reply: reply.value.trim() },
  };
  reset();
  document.getSelection()?.removeAllRanges();
  document.dispatchEvent(new CustomEvent(QUOTE_BLOCK_EVENT, { detail }));
}

/** Enter adds the block; Escape drops it, and goes no further — it must not
 *  also close the thread panel the reply sits in. */
function onReplyKeydown(e: KeyboardEvent): void {
  if (e.key === 'Escape') {
    e.stopPropagation();
    reset();
    return;
  }
  if (e.key !== 'Enter' || e.isComposing) return;
  e.preventDefault();
  commit();
}

function onPointerDown(e: PointerEvent): void {
  const path = e.composedPath();
  if ([barRef.value, popRef.value].some((el) => el && path.includes(el))) return;
  pressed = true;
  if (mode.value !== 'idle') reset();
}

function onPointerUp(): void {
  if (!pressed) return;
  pressed = false;
  schedule(0);
}

function onSelectionChange(): void {
  if (!pressed && mode.value !== 'reply') schedule(SETTLE_MS);
}

function onViewportChange(): void {
  if (mode.value !== 'idle') place();
}

onMounted(() => {
  document.addEventListener('pointerdown', onPointerDown, true);
  document.addEventListener('pointerup', onPointerUp, true);
  document.addEventListener('pointercancel', onPointerUp, true);
  document.addEventListener('selectionchange', onSelectionChange);
  document.addEventListener('scroll', onViewportChange, { capture: true, passive: true });
  globalThis.addEventListener('resize', onViewportChange);
});

onBeforeUnmount(() => {
  clearTimeout(timer);
  document.removeEventListener('pointerdown', onPointerDown, true);
  document.removeEventListener('pointerup', onPointerUp, true);
  document.removeEventListener('pointercancel', onPointerUp, true);
  document.removeEventListener('selectionchange', onSelectionChange);
  document.removeEventListener('scroll', onViewportChange, { capture: true });
  globalThis.removeEventListener('resize', onViewportChange);
});
</script>

<template>
  <Teleport to="body">
    <!-- Pressing the bar must not take focus or the selection from the reply. -->
    <div
      v-if="mode === 'bar'"
      ref="barRef"
      class="quote-bar"
      :style="style"
      @pointerdown.prevent
      @mousedown.prevent
    >
      <button type="button" class="quote-bar__btn" @click="openReply">
        <Quote :size="16" aria-hidden="true" />
        Quote
      </button>
    </div>
    <div
      v-else-if="mode === 'reply'"
      ref="popRef"
      class="quote-pop"
      role="dialog"
      aria-label="Reply to the quote"
      :style="style"
    >
      <p class="quote-pop__quote">{{ quote }}</p>
      <input
        ref="inputRef"
        v-model="reply"
        class="quote-pop__input"
        type="text"
        aria-label="Reply"
        placeholder="Reply…"
        @keydown="onReplyKeydown"
      />
    </div>
  </Teleport>
</template>

<style scoped lang="scss">
.quote-bar,
.quote-pop {
  position: fixed;
  z-index: 1150;
  padding: var(--space-xs);
  border-radius: 8px;
  background: var(--surface);
  animation: fade var(--dur-1) var(--ease-out);
}

.quote-bar__btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  height: var(--control-h);
  padding: 0 12px;
  border-radius: 6px;
  background: none;
  color: var(--text);
  font-weight: 600;
  cursor: pointer;
  transition: background-color var(--dur-1) var(--ease-out);

  &:hover {
    background: var(--surface-2);
  }
}

.quote-pop {
  display: flex;
  flex-direction: column;
  gap: var(--space-sm);
  width: min(360px, calc(100vw - 16px));
  padding: var(--space-sm);
}

// The quote cut to two lines, under a rule down its left drawn as a fill.
.quote-pop__quote {
  display: -webkit-box;
  -webkit-line-clamp: 2;
  line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
  margin: 0;
  padding-left: 10px;
  background: linear-gradient(var(--line), var(--line)) left / 2px 100% no-repeat;
  color: var(--muted);
  white-space: pre-wrap;
  overflow-wrap: break-word;
  line-height: 1.45;
}

.quote-pop__input {
  width: 100%;
}
</style>
