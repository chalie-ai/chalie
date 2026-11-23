<script lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue';

// Shared across every mounted dock. The footer dock and an open thread's reply
// dock both render at once, so the focus-routed handlers (voice transcript,
// interrupt-restore) and the pending-attachment strip act on the dock the user
// is composing in — otherwise one voice transcript would paste into both.
const activeDockKey = ref<string>('main');
</script>

<script setup lang="ts">
/**
 * InputDock — compose / send / stop, plus the full peripheral controls.
 *
 * WS single-owner rule: send/stop go through the session store; this component
 * never touches the WebSocket directly.
 */
import { storeToRefs } from 'pinia';
import { ConfigType, webPlatformAdapter } from '@chalie/shared';
import { readDomContext } from '../../utils/domContext';
import { on } from '../../composables/useEventBus';
import { useSessionStore } from '../../stores/session';
import { useVoiceStore } from '../../stores/voice';
import { useAttachmentsStore } from '../../stores/attachments';
import { useContextUsageStore } from '../../stores/contextUsage';
import { lsGet, lsSet } from '../../utils/storage';
import { providers, system } from '../../api';
import type { Provider } from '../../api/providers';
import ImageAttachStrip from '../upload/ImageAttachStrip.vue';
import QueuedMessages from '../conversation/QueuedMessages.vue';
import OffScreenAsks from './OffScreenAsks.vue';
import DockError from './DockError.vue';
import { Plus, Mic, Send, ChevronDown, ChevronLeft, ChevronRight } from '@lucide/vue';

/**
 * `turnId` is the only thing that distinguishes a thread reply from a main-dock
 * send: set, it appends to that thread; null (the dock default), the chat API
 * scaffolds a new thread. The footer dock is fixed; the thread panel's reply
 * dock sets `turnId` and renders in-flow at the foot of the panel.
 */
const props = withDefaults(defineProps<{ dockId: string; turnId?: number | null; type?: string }>(), {
  turnId: null,
  type: ConfigType.USER,
});

// Stable id for this dock so focus routing can tell the footer ('main') from a
// thread reply apart. Becomes the active dock on any interaction (focus/pointer).
const dockKey = props.turnId == null ? 'main' : `t${props.turnId}`;
const isActiveDock = computed(() => activeDockKey.value === dockKey);
function markActive(): void {
  activeDockKey.value = dockKey;
}

const session = useSessionStore();
const voiceStore = useVoiceStore();
const attachments = useAttachmentsStore();
const contextUsage = useContextUsageStore();
const { available: voiceAvailable, recorderState } = storeToRefs(voiceStore);
// The ring's % label shows on hover and keyboard focus; a phone has neither,
// so a tap toggles it.
const percentShown = ref(false);

const THINKING_ITEMS = [
  { level: 'auto', label: 'Auto' },
  { level: 'medium', label: 'Medium' },
  { level: 'high', label: 'High' },
] as const;

const textareaRef = ref<HTMLTextAreaElement | null>(null);
const sendBtnRef = ref<HTMLButtonElement | null>(null);
const text = ref('');

// Restore draft: persisted on every change so a reload/close/navigate recovers it.
// Keyed per thread so a reply draft never bleeds into the main composer.
const DRAFT_KEY = props.turnId == null ? 'chalie:draft' : `chalie:draft:t${props.turnId}`;
const stored = lsGet(DRAFT_KEY);
if (stored) text.value = stored;
watch(text, (v) => lsSet(DRAFT_KEY, v));

const fileInputRef = ref<HTMLInputElement | null>(null);

const footerRef = ref<HTMLElement | null>(null);

type Level = (typeof THINKING_ITEMS)[number]['level'];

// What the dock's next message runs on: its thread's model and level. The
// main dock's thread is the channel's latest, which a new thread copies; on a
// channel with no thread yet it shows the default and has nothing to change
// until the first message makes one.
const level = ref<Level>('auto');
const providerId = ref<number | null>(null);
const hasThread = ref(false);
const canPick = computed(() => props.turnId != null || hasThread.value);
// Every configured model; null until the list has been read.
const models = ref<Provider[] | null>(null);
// A model is shown by the model it runs; where two providers run the same one,
// each adds its provider's name so the two can be told apart.
const sharedModels = computed(() => {
  const seen = new Set<string>();
  const shared = new Set<string>();
  for (const m of models.value ?? []) (seen.has(m.model) ? shared : seen).add(m.model);
  return shared;
});
const modelName = (m: Provider): string => (sharedModels.value.has(m.model) ? `${m.model} (${m.name})` : m.model);
const levelLabel = (l: string): string => THINKING_ITEMS.find((i) => i.level === l)?.label ?? l;
const triggerLabel = computed(() => {
  const picked = models.value?.find((m) => m.id === providerId.value);
  return picked ? `${modelName(picked)} · ${levelLabel(level.value)}` : levelLabel(level.value);
});
// A null turnId (the footer dock) reads the channel's latest reading; a thread
// panel's dock reads its own thread. Both are fed by the same `context_usage`
// frame — the store resolves which entry that is. The reading is measured
// against the window of the model the next message goes to, as the server's
// context check measures it, so a switch to a smaller model shows at once.
const pickedWindow = computed(() => models.value?.find((m) => m.id === providerId.value)?.context_window ?? null);
const percentUsed = computed(() => contextUsage.percentUsedFor(props.type, props.turnId, pickedWindow.value));

// The menu drills in: it lists the models, and choosing one lists that
// model's levels (`menuModel`), with a row back to the models at the top.
const menuOpen = ref(false);
const menuModel = ref<Provider | null>(null);
const selectWrapRef = ref<HTMLDivElement | null>(null);
const triggerRef = ref<HTMLButtonElement | null>(null);
const menuRef = ref<HTMLDivElement | null>(null);

/** True when there is something to send: non-empty text OR ≥1 attachment.
 *  Mirrors the handleSend guard — both gate on getFiles(). */
const canSend = computed(() => text.value.trim().length > 0 || attachments.getFiles().length > 0);

async function handleSend(): Promise<void> {
  const trimmed = text.value.trim();
  const files = attachments.getFiles();

  // Guard here too so we don't clear needlessly. Same gate as canSend.
  if (!trimmed && files.length === 0) return;

  // Read the dock's DOM contract at click time — the ref can go null if the
  // dock unmounts across an await, and the send must target the same lane.
  const { turnId, type } = readDomContext(footerRef.value);

  // Clear textarea before awaiting so the UI feels instant.
  text.value = '';
  await nextTick();

  await session.sendMessage(trimmed, files, turnId, type);
  // A main-dock send may start the channel's first thread, which the menu
  // then has to change.
  if (props.turnId == null) void loadSettings();

  // sendMessage takes ownership of `files` in BOTH branches — a direct dispatch
  // uploads them; a busy send with files queues the whole {text, files}
  // (queue.ts stores and replays them on drain). Either way the strip must
  // clear: leaving files pending after a queued send re-attaches them to the
  // user's NEXT message, a duplicate upload.
  attachments.clear();

  textareaRef.value?.focus();
}

// Enter-to-send is a desktop-only affordance. On a device with a real keyboard
// (fine pointer + hover) Enter sends and Shift+Enter inserts a newline; on touch
// (mobile/tablet) Enter always inserts a newline and only the button sends. The
// keystroke routes through THIS dock's own send button — a real click on the
// element it was captured from — so it lands in the exact lane (main vs thread)
// it fired in and inherits the button's disabled/guard state for free. IME
// composition is never interrupted. Cancelling an in-flight turn stays the
// act-trail's stop/undo button, not the dock.
function onTextareaKeydown(e: KeyboardEvent): void {
  if (e.key !== 'Enter' || e.shiftKey || e.isComposing) return;
  const desktop = globalThis.matchMedia?.('(hover: hover) and (pointer: fine)').matches;
  if (!desktop) return;
  e.preventDefault();
  sendBtnRef.value?.click();
}

function openFilePicker(): void {
  fileInputRef.value?.click();
}

/** addFiles dispatches on type, so one input covers images and documents alike. */
function onFileInputChange(e: Event): void {
  const input = e.target as HTMLInputElement;
  if (input.files?.length) void attachments.addFiles(input.files);
  input.value = ''; // allow re-selecting the same file
}

/** Read what the dock's next message runs on: its thread's settings, -1
 *  naming the main dock's thread. */
async function loadSettings(): Promise<void> {
  try {
    const r = await system.threadSettings(props.type, props.turnId ?? -1);
    level.value = asLevel(r.thinking_level);
    providerId.value = r.provider_id;
    hasThread.value = r.turn_id != null;
  } catch {
    level.value = 'auto';
  }
}

async function loadModels(): Promise<void> {
  try {
    models.value = await providers.all();
  } catch {
    session.errorMessage = 'Could not load the models';
  }
}

function asLevel(v: string): Level {
  return v === 'medium' || v === 'high' ? v : 'auto';
}

/** Move focus into the menu once it renders: onto the checked row, else the first. */
function focusMenuRow(): void {
  void nextTick(() => {
    const rows = menuRef.value?.querySelectorAll<HTMLElement>('[role^="menuitem"]');
    const checked = menuRef.value?.querySelector<HTMLElement>('.active');
    (checked ?? rows?.[0])?.focus();
  });
}

function closeMenu(): void {
  menuOpen.value = false;
  menuModel.value = null;
}

function toggleMenu(): void {
  if (menuOpen.value) return closeMenu();
  menuOpen.value = true;
  focusMenuRow();
}

function openLevels(model: Provider): void {
  menuModel.value = model;
  focusMenuRow();
}

function backToModels(): void {
  menuModel.value = null;
  focusMenuRow();
}

/** Arrow keys move between rows, wrapping; Escape backs out of a model's
 *  levels, then closes. Enter picks through the row's own click. */
function onMenuKeydown(e: KeyboardEvent): void {
  if (e.key === 'Escape') {
    e.stopPropagation();
    if (menuModel.value) return backToModels();
    closeMenu();
    triggerRef.value?.focus();
    return;
  }
  if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return;
  e.preventDefault();
  const rows = Array.from(menuRef.value?.querySelectorAll<HTMLElement>('[role^="menuitem"]') ?? []);
  const at = rows.indexOf(document.activeElement as HTMLElement);
  const step = e.key === 'ArrowDown' ? 1 : -1;
  rows[(at + step + rows.length) % rows.length]?.focus();
}

/** Save the pick to the dock's thread at once, -1 naming the main dock's
 *  thread; a failed save keeps the old settings. */
async function selectLevel(next: Level): Promise<void> {
  const model = menuModel.value;
  closeMenu();
  triggerRef.value?.focus();
  if (!model) return;
  const saved = await system
    .saveThreadSettings(props.type, props.turnId ?? -1, { provider_id: model.id, thinking_level: next })
    .catch(() => null);
  if (!saved) {
    session.errorMessage = 'Could not change the model or thinking level';
    return;
  }
  providerId.value = saved.provider_id;
  level.value = asLevel(saved.thinking_level);
}

/** Where a click landed is read from its path, fixed when it was dispatched:
 *  choosing a model or going back re-renders the menu before the click reaches
 *  the document, so the row clicked has already left the page. */
function onDocumentClick(e: MouseEvent): void {
  const wrap = selectWrapRef.value;
  if (menuOpen.value && !(wrap && e.composedPath().includes(wrap))) closeMenu();
}

// Raw File objects are structurally non-persistable, so the only loss the
// draft-restoration above doesn't cover is a discard of pending attachments.
// Warn on beforeunload so an accidental reload/close/navigate confirms first.
function onBeforeUnload(e: BeforeUnloadEvent): void {
  if (attachments.getFiles().length > 0) {
    e.preventDefault();
    e.returnValue = '';
  }
}

/** A turn was stopped/undone — restore its draft into the dock that owns its
 *  scope. Matched by turn_id (like the sibling `chalie:edit-queued` handler),
 *  not `isActiveDock`: the stop control lives in the turn's activity line,
 *  never in an InputDock footer, so `activeDockKey` never points at the right
 *  dock. */
function onTurnInterrupted(e: Event): void {
  const detail = (e as CustomEvent<{ text: string; turnId: number | null }>).detail;
  if (detail.turnId !== props.turnId) return;
  text.value = detail.text ?? '';
  nextTick(() => {
    textareaRef.value?.focus();
    // Move cursor to end.
    const el = textareaRef.value;
    if (el) {
      el.selectionStart = el.selectionEnd = el.value.length;
    }
  });
}

/** A queued message was clicked for editing — route it to the dock that owns its
 *  scope (footer for the spine, the panel dock for a thread), append it to any
 *  draft, and focus. Matched by turn_id, not active state, so a spine click lands
 *  in the footer even while a thread dock is focused. */
function onEditQueued(e: Event): void {
  const detail = (e as CustomEvent<{ turnId: number | null; text: string }>).detail;
  if (detail.turnId !== props.turnId) return;
  markActive();
  text.value = text.value ? `${text.value}\n${detail.text}` : detail.text;
  nextTick(() => {
    textareaRef.value?.focus();
    const el = textareaRef.value;
    if (el) el.selectionStart = el.selectionEnd = el.value.length;
  });
}

// The footer dock is fixed to the viewport, so .conversation-spine can't see
// its height to reserve space beneath the last message. The attachment strip
// and queued-message chips grow the dock UPWARD, and a static reserve would let
// that growth overlap the act-trail (the queued-messages defect). Publish the
// dock's live height as --dock-height on :root so the spine's bottom padding
// tracks it. Footer dock only — the inline thread reply dock is in-flow (static)
// and needs no reserve.
let _dockResizeObserver: ResizeObserver | null = null;

let _unsubVoiceTranscript: (() => void) | null = null;

/** Paste the voice transcript into the compose textarea for review — does NOT auto-send. */
function onVoiceTranscript({ text: transcript }: { text: string }): void {
  if (!isActiveDock.value) return;
  text.value = transcript;
  nextTick(() => {
    textareaRef.value?.focus();
    // Move cursor to end.
    const el = textareaRef.value;
    if (el) {
      el.selectionStart = el.selectionEnd = el.value.length;
    }
  });
}

onMounted(() => {
  document.addEventListener('session:turn-interrupted', onTurnInterrupted);
  document.addEventListener('chalie:edit-queued', onEditQueued);
  _unsubVoiceTranscript = on('chalie:voice-transcript', onVoiceTranscript);
  document.addEventListener('click', onDocumentClick);
  globalThis.addEventListener('beforeunload', onBeforeUnload);

  // Reserve the spine's bottom space against this dock's LIVE height (footer only).
  if (props.turnId == null && footerRef.value) {
    _dockResizeObserver = new ResizeObserver((entries) => {
      const box = entries[0]?.borderBoxSize?.[0];
      const h = box ? box.blockSize : (entries[0]?.contentRect.height ?? 0);
      if (h > 0) document.documentElement.style.setProperty('--dock-height', `${Math.ceil(h)}px`);
    });
    _dockResizeObserver.observe(footerRef.value);
  }

  void loadSettings();
  void loadModels();
});

onBeforeUnmount(() => {
  document.removeEventListener('session:turn-interrupted', onTurnInterrupted);
  document.removeEventListener('chalie:edit-queued', onEditQueued);
  _unsubVoiceTranscript?.();
  document.removeEventListener('click', onDocumentClick);
  globalThis.removeEventListener('beforeunload', onBeforeUnload);
  if (props.turnId == null) {
    _dockResizeObserver?.disconnect();
    _dockResizeObserver = null;
    document.documentElement.style.removeProperty('--dock-height');
  }
  // Hand focus routing back to the footer when an inline dock collapses.
  if (activeDockKey.value === dockKey) activeDockKey.value = 'main';
  // The recorder is a shared singleton — only the permanent footer dock tears it
  // down, so collapsing a thread mid-recording can't kill it under the footer.
  if (props.turnId == null) voiceStore.destroyRecorder();
});
</script>

<template>
  <footer
    ref="footerRef"
    :id="dockId"
    class="input-dock"
    :class="{ 'input-dock--inline': turnId != null }"
    :data-turn-id="turnId"
    :data-type="type"
    @focusin="markActive"
    @pointerdown="markActive"
  >
    <DockError />

    <!-- Pending asks no live line on screen shows; active dock only, like the attach strip. -->
    <OffScreenAsks v-if="isActiveDock" />

    <!-- Attachments are a shared store; render the pending strip only in the
         active dock so the footer and an open thread don't show duplicates. -->
    <ImageAttachStrip v-if="isActiveDock" />

    <!-- Pending (queued) sends for this dock's scope, floating above the composer. -->
    <QueuedMessages :thread-id="turnId" />

    <div class="input-dock__outer">
      <div class="input-dock__inner">
        <button
          v-if="voiceAvailable"
          id="voiceRecBtn"
          class="btn-icon voice-rec-btn"
          aria-label="Record voice message"
          :data-state="recorderState"
          @click="voiceStore.toggleRecording()"
        >
          <Mic class="voice-rec-btn__mic" :size="16" />
          <span class="voice-rec-btn__dot" aria-hidden="true"></span>
          <span class="voice-rec-btn__spinner" aria-hidden="true"></span>
        </button>

        <button
          id="attachBtn"
          class="btn-action btn-action--attach"
          aria-label="Attach"
          @click="openFilePicker"
        >
          <Plus :size="16" />
        </button>

        <textarea
          id="chatInput"
          ref="textareaRef"
          v-model="text"
          class="input-dock__textarea"
          :aria-label="turnId != null ? 'Reply in thread' : 'Message Chalie'"
          :placeholder="turnId != null ? 'Reply in this thread…' : 'Talk to Chalie…'"
          rows="1"
          @keydown="onTextareaKeydown"
        ></textarea>

        <button
          ref="sendBtnRef"
          class="btn-action btn-action--send"
          aria-label="Send message"
          :disabled="!canSend"
          @click="handleSend()"
        >
          <Send :size="16" />
        </button>
      </div>
    </div>

    <div class="dock-controls">
      <div ref="selectWrapRef" class="model-select">
        <!-- With no model configured there is nothing to pick: the trigger
             opens the Brain, whose first page adds one. -->
        <button
          v-if="models?.length === 0"
          id="modelTrigger"
          class="model-select__trigger"
          type="button"
          @click="webPlatformAdapter.openBrain()"
        >
          <span class="model-select__swirl" aria-hidden="true"></span>
          <span class="model-select__value">Add a model</span>
        </button>
        <button
          v-else
          id="modelTrigger"
          ref="triggerRef"
          class="model-select__trigger"
          type="button"
          aria-haspopup="menu"
          :aria-expanded="menuOpen"
          :disabled="!canPick"
          @click.stop="toggleMenu"
        >
          <span class="model-select__swirl" aria-hidden="true"></span>
          <span id="modelLabel" class="model-select__value">{{ triggerLabel }}</span>
          <ChevronDown :size="16" style="opacity: 0.6" />
        </button>
        <div
          v-if="menuOpen"
          id="modelMenu"
          ref="menuRef"
          class="model-select__menu"
          role="menu"
          :aria-label="menuModel ? `${modelName(menuModel)} thinking levels` : 'Models'"
          @keydown="onMenuKeydown"
        >
          <template v-if="menuModel">
            <button class="model-select__item model-select__item--back" type="button" role="menuitem" @click="backToModels">
              <ChevronLeft :size="16" aria-hidden="true" />
              <span class="model-select__name">{{ modelName(menuModel) }}</span>
            </button>
            <button
              v-for="item in THINKING_ITEMS"
              :key="item.level"
              class="model-select__item"
              :class="{ active: menuModel.id === providerId && level === item.level }"
              :data-level="item.level"
              type="button"
              role="menuitemradio"
              :aria-checked="menuModel.id === providerId && level === item.level"
              @click="selectLevel(item.level)"
            >
              {{ item.label }}
            </button>
          </template>
          <template v-else>
            <button
              v-for="model in models"
              :key="model.id"
              class="model-select__item"
              :class="{ active: model.id === providerId }"
              :data-model="model.id"
              type="button"
              role="menuitem"
              aria-haspopup="menu"
              @click="openLevels(model)"
            >
              <span class="model-select__name">{{ modelName(model) }}</span>
              <ChevronRight :size="16" aria-hidden="true" />
            </button>
          </template>
        </div>
      </div>
      <div v-if="percentUsed != null" class="context-ring">
        <div
          id="contextDisplay"
          class="context-ring__meter"
          :class="{ 'context-ring--open': percentShown }"
          role="meter"
          aria-label="Context used"
          aria-valuemin="0"
          aria-valuemax="100"
          :aria-valuenow="percentUsed"
        >
          <svg class="context-ring__dial" viewBox="0 0 36 36" aria-hidden="true">
            <circle class="context-ring__track" cx="18" cy="18" r="14" />
            <circle
              class="context-ring__fill"
              cx="18"
              cy="18"
              r="14"
              pathLength="100"
              stroke-dasharray="100"
              :stroke-dashoffset="100 - percentUsed"
            />
          </svg>
          <span class="context-ring__label">{{ percentUsed }}%</span>
        </div>
        <button
          type="button"
          class="context-ring__toggle"
          aria-label="Show the percentage"
          :aria-pressed="percentShown"
          @click="percentShown = !percentShown"
        />
      </div>
    </div>

    <!-- No accept attr: any file may be attached, the same as dropping one.
         No capture attr: lets mobile show the standard OS picker (library +
         take-photo), WhatsApp-style. -->
    <input
      id="attachFileInput"
      ref="fileInputRef"
      type="file"
      aria-label="Attach file"
      multiple
      hidden
      @change="onFileInputChange"
    />
  </footer>
</template>
