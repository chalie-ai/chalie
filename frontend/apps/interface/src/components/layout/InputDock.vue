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
import { ConfigType } from '@chalie/shared';
import { readDomContext } from '../../utils/domContext';
import { on } from '../../composables/useEventBus';
import { useSessionStore } from '../../stores/session';
import { useVoiceStore } from '../../stores/voice';
import { useAttachmentsStore } from '../../stores/attachments';
import { useContextUsageStore } from '../../stores/contextUsage';
import { useAmbientSensor } from '../../composables/useAmbientSensor';
import { lsGet, lsSet } from '../../utils/storage';
import { system } from '../../api';
import ImageAttachStrip from '../upload/ImageAttachStrip.vue';
import QueuedMessages from '../conversation/QueuedMessages.vue';
import { Plus, Mic, Send, X, AlertTriangle, ChevronDown } from '@lucide/vue';

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
const ambient = useAmbientSensor();
const { available: voiceAvailable, recorderState } = storeToRefs(voiceStore);
// A null turnId (the footer dock) reads the channel's latest reading; a thread
// panel's dock reads its own thread. Both are fed by the same `context_usage`
// frame — the store resolves which entry that is.
const usageDisplay = computed(() => contextUsage.usageDisplayFor(props.type, props.turnId));
const usageRatio = computed(() => contextUsage.usageRatioFor(props.type, props.turnId));
// The meter shows the percentage used ("30%"); the raw "X/Y" token counts move
// to the hover tooltip (see the container's :title below) so the inline chip
// stays compact.
const usagePercent = computed(() => Math.round(usageRatio.value * 100) + '%');

const THINKING_ITEMS = [
  { level: 'auto', label: 'Auto' },
  { level: 'medium', label: 'Medium' },
  { level: 'high', label: 'High' },
] as const;

/** Union of every attachable kind — one field, images and documents alike. */
const ATTACH_ACCEPT =
  'image/jpeg,image/png,image/webp,image/gif,.pdf,.docx,.pptx,.html,.htm,.txt,.md,.csv,.json,.xml';

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

const thinkingMenuOpen = ref(false);
const thinkingWrapRef = ref<HTMLDivElement | null>(null);
const footerRef = ref<HTMLElement | null>(null);

const level = ref<'auto' | 'medium' | 'high'>('auto');
const levelLabel = computed(() => level.value.charAt(0).toUpperCase() + level.value.slice(1));

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

  await session.sendMessage(trimmed, files, turnId, type, level.value);

  // sendMessage takes ownership of `files` in BOTH branches — a direct dispatch
  // uploads them; a busy send queues the whole {text, files} (queue.ts stores
  // and replays them on drain). Either way the strip must clear: leaving files
  // pending after a queued send re-attaches them to the user's NEXT message,
  // a duplicate upload.
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
  if (input.files?.length) attachments.addFiles(input.files);
  input.value = ''; // allow re-selecting the same file
}

function selectLevel(next: (typeof THINKING_ITEMS)[number]['level']): void {
  level.value = next;
  thinkingMenuOpen.value = false;
}

function onDocumentClick(e: MouseEvent): void {
  const target = e.target as Node;
  if (thinkingMenuOpen.value && !thinkingWrapRef.value?.contains(target)) {
    thinkingMenuOpen.value = false;
  }
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
 *  not `isActiveDock`: the stop control lives in ActCycle/TurnView, never in
 *  an InputDock footer, so `activeDockKey` never points at the right dock. */
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

  // Behavioral signals: typing cadence feeds the ambient snapshot.
  if (textareaRef.value) ambient.bindTypingInput(textareaRef.value);

  system.thinkingLevel(props.type, props.turnId ?? -1).then((r) => {
    const v = r?.level;
    level.value = v === 'medium' || v === 'high' ? v : 'auto';
  }).catch(() => { level.value = 'auto'; });
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
    <div v-if="session.errorMessage" class="dock-error" role="alert">
      <AlertTriangle class="dock-error__icon" :size="18" aria-hidden="true" />
      <span class="dock-error__text">{{ session.errorMessage }}</span>
      <button
        class="dock-error__close"
        type="button"
        aria-label="Dismiss error"
        @click="session.errorMessage = null"
      >
        <X :size="16" />
      </button>
    </div>

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
          <Mic class="voice-rec-btn__mic" :size="18" />
          <span class="voice-rec-btn__dot" aria-hidden="true"></span>
          <span class="voice-rec-btn__spinner" aria-hidden="true"></span>
        </button>

        <button
          id="attachBtn"
          class="btn-action btn-action--attach"
          aria-label="Attach"
          @click="openFilePicker"
        >
          <Plus :size="20" />
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
          <Send :size="20" />
        </button>
      </div>
    </div>

    <div class="dock-controls">
      <div ref="thinkingWrapRef" class="thinking-select">
        <button
          id="thinkingTrigger"
          class="thinking-select__trigger"
          type="button"
          aria-haspopup="true"
          :aria-expanded="thinkingMenuOpen"
          @click.stop="thinkingMenuOpen = !thinkingMenuOpen"
        >
          <span class="thinking-select__swirl" aria-hidden="true"></span>
          <span id="thinkingLabel" class="thinking-select__value">{{ levelLabel }}</span>
          <ChevronDown :size="12" style="opacity: 0.6" />
        </button>
        <div
          id="thinkingMenu"
          class="thinking-select__menu"
          :class="{ hidden: !thinkingMenuOpen }"
          role="menu"
        >
          <button
            v-for="item in THINKING_ITEMS"
            :key="item.level"
            class="thinking-select__item"
            :class="{ active: level === item.level }"
            :data-level="item.level"
            type="button"
            role="menuitem"
            @click="selectLevel(item.level)"
          >
            {{ item.label }}
          </button>
        </div>
      </div>
      <div
        id="contextDisplay"
        class="context-display"
        :class="{ hidden: !usageDisplay }"
        :title="`${usageDisplay} context`"
      >
        <span class="context-indicator">
          <b>{{ usagePercent }}</b>
        </span>
        <span class="meter"><i :style="{ width: usageRatio * 100 + '%' }"></i></span>
      </div>
    </div>

    <!-- No capture attr: lets mobile show the standard OS picker (library +
         take-photo), WhatsApp-style. -->
    <input
      id="attachFileInput"
      ref="fileInputRef"
      type="file"
      :accept="ATTACH_ACCEPT"
      aria-label="Attach file"
      multiple
      hidden
      @change="onFileInputChange"
    />
  </footer>
</template>
