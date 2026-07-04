<script setup lang="ts">
/**
 * InputDock — compose / send / stop, plus the full peripheral controls.
 *
 * WS single-owner rule: send/stop go through the session store; this component
 * never touches the WebSocket directly.
 */
import { ref, computed, onMounted, onBeforeUnmount, nextTick, watch } from 'vue';
import { storeToRefs } from 'pinia';
import { on } from '../../composables/useEventBus';
import { useSessionStore } from '../../stores/session';
import { useVoiceStore } from '../../stores/voice';
import { useAttachmentsStore } from '../../stores/attachments';
import { useContextUsageStore } from '../../stores/contextUsage';
import { useAmbientSensor } from '../../composables/useAmbientSensor';
import { system } from '../../api/system';
import { lsGet, lsSet } from '../../utils/storage';
import type { AttachmentPreview as ConvoAttachmentPreview } from '../../stores/conversation';
import ImageAttachStrip from '../upload/ImageAttachStrip.vue';
import { FileText, Image, Plus, Mic, Send, X, AlertTriangle, LoaderCircle } from '@lucide/vue';

const session = useSessionStore();
const voiceStore = useVoiceStore();
const attachments = useAttachmentsStore();
const contextUsage = useContextUsageStore();
const ambient = useAmbientSensor();

const { available: voiceAvailable, recorderState } = storeToRefs(voiceStore);
const { level, levelLabel, usageDisplay } = storeToRefs(contextUsage);

const THINKING_ITEMS = [
  { level: 'auto', label: 'Auto' },
  { level: 'medium', label: 'Medium' },
  { level: 'high', label: 'High' },
] as const;

const textareaRef = ref<HTMLTextAreaElement | null>(null);
const text = ref('');

// Restore draft: persisted on every change so a reload/close/navigate recovers it.
const DRAFT_KEY = 'chalie:draft';
const stored = lsGet(DRAFT_KEY);
if (stored) text.value = stored;
watch(text, (v) => lsSet(DRAFT_KEY, v));

const attachMenuOpen = ref(false);
const attachBtnRef = ref<HTMLButtonElement | null>(null);
const attachMenuRef = ref<HTMLDivElement | null>(null);
const imageInputRef = ref<HTMLInputElement | null>(null);
const docInputRef = ref<HTMLInputElement | null>(null);
/** Hidden when no vision-capable provider is configured. */
const hasVisionProvider = ref(true);

const thinkingMenuOpen = ref(false);
const thinkingWrapRef = ref<HTMLDivElement | null>(null);

/** True when there is something to send: non-empty text OR ≥1 attachment.
 *  Mirrors the handleSend guard — both gate on getFiles(). */
const canSend = computed(
  () => text.value.trim().length > 0 || attachments.getFiles().length > 0,
);

/** Auto-resize: reset height then cap at 120px. */
function grow(): void {
  const el = textareaRef.value;
  if (!el) return;
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}

async function handleSend(): Promise<void> {
  const trimmed = text.value.trim();

  // Read files + previews BEFORE clear() wipes the strip.
  const files = attachments.getFiles();
  const previews: ConvoAttachmentPreview[] = attachments.previews.map((p) => ({
    filename: p.filename,
    objectUrl: p.dataUrl,
    isImage: p.isImage,
  }));

  // Guard here too so we don't clear/re-grow needlessly. Same gate as canSend.
  if (!trimmed && files.length === 0) return;

  // Clear the image strip only on a fresh turn, not the mid-ACT append path —
  // so capture send-mode before the store flips isSending.
  const wasSending = session.isSending;

  // Clear textarea before awaiting so the UI feels instant.
  text.value = '';
  await nextTick();
  grow();

  await session.sendMessage(trimmed, 'text', files, previews);

  if (!wasSending) attachments.clear();

  textareaRef.value?.focus();
}

// Multi-line textarea: Enter inserts a newline and ONLY the send button submits,
// so there is no keydown handler. Cancelling an in-flight turn is the act-trail's
// stop/undo button, not the dock.

function chooseDocument(): void {
  attachMenuOpen.value = false;
  docInputRef.value?.click();
}

function chooseImage(): void {
  attachMenuOpen.value = false;
  imageInputRef.value?.click();
}

/** Shared by both pickers — addFiles dispatches on type. */
function onFileInputChange(e: Event): void {
  const input = e.target as HTMLInputElement;
  if (input.files?.length) void attachments.addFiles(input.files);
  input.value = ''; // allow re-selecting the same file
}

function selectLevel(next: (typeof THINKING_ITEMS)[number]['level']): void {
  void contextUsage.setLevel(next);
  thinkingMenuOpen.value = false;
}

function onDocumentClick(e: MouseEvent): void {
  const target = e.target as Node;
  if (
    attachMenuOpen.value &&
    !attachMenuRef.value?.contains(target) &&
    !attachBtnRef.value?.contains(target)
  ) {
    attachMenuOpen.value = false;
  }
  if (thinkingMenuOpen.value && !thinkingWrapRef.value?.contains(target)) {
    thinkingMenuOpen.value = false;
  }
}

function onWindowScroll(): void {
  if (attachMenuOpen.value) attachMenuOpen.value = false;
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

function onTurnInterrupted(e: Event): void {
  const detail = (e as CustomEvent<{ text: string }>).detail;
  text.value = detail.text ?? '';
  nextTick(() => {
    grow();
    textareaRef.value?.focus();
    // Move cursor to end.
    const el = textareaRef.value;
    if (el) {
      el.selectionStart = el.selectionEnd = el.value.length;
    }
  });
}

let _unsubVoiceTranscript: (() => void) | null = null;

/** Paste the voice transcript into the compose textarea for review — does NOT auto-send. */
function onVoiceTranscript({ text: transcript }: { text: string }): void {
  text.value = transcript;
  nextTick(() => {
    grow();
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
  _unsubVoiceTranscript = on('chalie:voice-transcript', onVoiceTranscript);
  document.addEventListener('click', onDocumentClick);
  globalThis.addEventListener('scroll', onWindowScroll, { passive: true });
  globalThis.addEventListener('beforeunload', onBeforeUnload);

  // Behavioral signals: typing cadence feeds the ambient snapshot.
  if (textareaRef.value) ambient.bindTypingInput(textareaRef.value);

  void contextUsage.loadLevel();
  void contextUsage.refresh();

  // Vision-provider gating for the image attach option.
  system
    .authStatus()
    .then((s) => {
      hasVisionProvider.value = s.has_vision_provider;
    })
    .catch(() => {
      /* leave the image option visible on error */
    });
});

onBeforeUnmount(() => {
  document.removeEventListener('session:turn-interrupted', onTurnInterrupted);
  _unsubVoiceTranscript?.();
  document.removeEventListener('click', onDocumentClick);
  globalThis.removeEventListener('scroll', onWindowScroll);
  globalThis.removeEventListener('beforeunload', onBeforeUnload);
  voiceStore.destroyRecorder();
});
</script>

<template>
  <footer class="input-dock">
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

    <ImageAttachStrip />

    <div
      id="attachMenu"
      ref="attachMenuRef"
      class="attach-menu"
      :class="{ hidden: !attachMenuOpen }"
    >
      <div class="attach-menu__inner">
        <button class="attach-menu__item" type="button" @click="chooseDocument">
          <FileText :size="16" />
          <span>Attach Document</span>
        </button>
        <button
          v-if="hasVisionProvider"
          id="attachImageBtn"
          class="attach-menu__item"
          type="button"
          @click="chooseImage"
        >
          <Image :size="16" />
          <span>Take Photo / Pick Image</span>
        </button>
      </div>
    </div>

    <div class="input-dock__outer">
      <div class="input-dock__inner">
        <button
          id="attachBtn"
          ref="attachBtnRef"
          class="btn-action btn-action--attach"
          :class="{ active: attachMenuOpen }"
          aria-label="Attach"
          @click.stop="attachMenuOpen = !attachMenuOpen"
        >
          <Plus :size="20" />
        </button>

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

        <textarea
          id="chatInput"
          ref="textareaRef"
          v-model="text"
          class="input-dock__textarea"
          placeholder="Talk to Chalie..."
          rows="1"
          @input="grow"
        ></textarea>

        <button
          class="btn-action btn-action--send"
          :aria-label="session.isSending ? 'Sending...' : 'Send message'"
          :disabled="!canSend || session.isSending"
          @click="handleSend()"
        >
          <LoaderCircle v-if="session.isSending" class="btn-action--send__spinner" :size="20" />
          <Send v-else :size="20" />
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
          <span class="thinking-select__caption">Thinking</span>
          <span id="thinkingLabel" class="thinking-select__value">{{ levelLabel }}</span>
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
      <div id="contextDisplay" class="context-display" :class="{ hidden: !usageDisplay }">
        <span class="context-display__caption">Context</span>
        <span class="context-indicator" title="Last request size / context window">{{ usageDisplay }}</span>
      </div>
    </div>

    <!-- No capture attr: lets mobile show the standard OS picker (library +
         take-photo), WhatsApp-style. -->
    <input
      id="imageFileInput"
      ref="imageInputRef"
      type="file"
      accept="image/jpeg,image/png,image/webp,image/gif"
      aria-label="Attach image"
      multiple
      hidden
      @change="onFileInputChange"
    />

    <input
      id="docFileInput"
      ref="docInputRef"
      type="file"
      accept=".pdf,.docx,.pptx,.html,.htm,.txt,.md,.csv,.json,.xml"
      aria-label="Attach document"
      multiple
      hidden
      @change="onFileInputChange"
    />
  </footer>
</template>
