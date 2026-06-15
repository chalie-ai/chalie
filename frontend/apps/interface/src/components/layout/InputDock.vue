<script setup lang="ts">
/**
 * InputDock — compose / send / stop, plus the full peripheral controls.
 *
 * Port sources:
 *   legacy app.js   _initInput (auto-grow, send-on-click, onSendComplete focus),
 *                   _initAttachMenu (+ button, vision-provider gating),
 *                   _ambientSensor.bindTypingInput
 *   legacy chat.js  sendMessage (image files + previews; orchestration is in
 *                   the session store)
 *   legacy chat_controls.js thinking-level dropdown + context-usage indicator
 *   legacy voice_recorder.js mic button (toggle recording; state + error)
 *   legacy index.html .input-dock markup (classes preserved for interface.scss)
 *
 * WS single-owner rule is honoured: send/stop go through the session store;
 * this component never touches the WebSocket directly.
 */
import { ref, computed, onMounted, onBeforeUnmount, nextTick } from 'vue';
import { storeToRefs } from 'pinia';
import { on } from '../../composables/useEventBus';
import { useSessionStore } from '../../stores/session';
import { useVoiceStore } from '../../stores/voice';
import { useAttachmentsStore } from '../../stores/attachments';
import { useContextUsageStore } from '../../stores/contextUsage';
import { useAmbientSensor } from '../../composables/useAmbientSensor';
import { system } from '../../api/system';
import type { AttachmentPreview as ConvoAttachmentPreview } from '../../stores/conversation';
import ImageAttachStrip from '../upload/ImageAttachStrip.vue';

// ── Stores / composables ────────────────────────────────────────────────────

const session = useSessionStore();
const voiceStore = useVoiceStore();
const attachments = useAttachmentsStore();
const contextUsage = useContextUsageStore();
const ambient = useAmbientSensor();

const { isSending } = storeToRefs(session);
const { available: voiceAvailable, recorderState, recError } = storeToRefs(voiceStore);
const { level, levelLabel, usageDisplay } = storeToRefs(contextUsage);

// Thinking-level options — order + labels match legacy index.html lines 249-251.
const THINKING_ITEMS = [
  { level: 'auto', label: 'Auto' },
  { level: 'medium', label: 'Medium' },
  { level: 'high', label: 'High' },
] as const;

// ── Local state ───────────────────────────────────────────────────────────────

const textareaRef = ref<HTMLTextAreaElement | null>(null);
const text = ref('');

// Attach-menu UI
const attachMenuOpen = ref(false);
const attachBtnRef = ref<HTMLButtonElement | null>(null);
const attachMenuRef = ref<HTMLDivElement | null>(null);
const imageInputRef = ref<HTMLInputElement | null>(null);
const docInputRef = ref<HTMLInputElement | null>(null);
/** Hidden when no vision-capable provider is configured (legacy app.js:699-706). */
const hasVisionProvider = ref(true);

// Thinking-level dropdown UI
const thinkingMenuOpen = ref(false);
const thinkingWrapRef = ref<HTMLDivElement | null>(null);

/** True when there is something to send: non-empty text OR ≥1 attachment
 *  (image OR document). Mirrors the handleSend guard exactly — both gate on
 *  getFiles(), the files that actually ride the multipart /chat request. */
const canSend = computed(
  () => text.value.trim().length > 0 || attachments.getFiles().length > 0,
);

// ── Auto-grow ─────────────────────────────────────────────────────────────────

/** Port of app.js _initInput auto-resize: reset height then cap at 120px. */
function grow(): void {
  const el = textareaRef.value;
  if (!el) return;
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}

// ── Send / Stop ───────────────────────────────────────────────────────────────

async function handleSend(): Promise<void> {
  const trimmed = text.value.trim();

  // Read files + previews BEFORE clear() wipes the strip (chat.js:132-136).
  // Images AND documents both ride the multipart /chat and render in the bubble.
  const files = attachments.getFiles();
  const previews: ConvoAttachmentPreview[] = attachments.previews.map((p) => ({
    filename: p.filename,
    objectUrl: p.dataUrl,
    isImage: p.isImage,
  }));

  // session.sendMessage no-ops on empty input, but guard here too so we don't
  // clear the textarea or re-grow unnecessarily (chat.js:138). Same gate as canSend.
  if (!trimmed && files.length === 0) return;

  // Capture send-mode before the store flips isSending: legacy clears the image
  // strip only on a fresh turn, not on the mid-ACT append path (chat.js:160).
  const wasSending = session.isSending;

  // Clear textarea before awaiting so the UI feels instant.
  text.value = '';
  await nextTick();
  grow();

  await session.sendMessage(trimmed, 'text', files, previews);

  if (!wasSending) attachments.clear();

  // Re-focus after the turn completes (port of app.js onSendComplete handler).
  textareaRef.value?.focus();
}

async function handleStop(): Promise<void> {
  await session.requestStop();
  // requestStop dispatches session:turn-interrupted which the listener below
  // catches to restore text; focus is handled there.
}

// ── Keyboard ──────────────────────────────────────────────────────────────────

function handleKeydown(e: KeyboardEvent): void {
  // Enter without Shift → send. Shift+Enter → newline (default behaviour).
  // While a turn is in flight, Enter still sends: session.sendMessage appends
  // the text to the active turn (the mid-ACT append is handled in the store).
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    handleSend();
  }
}

// ── Mic (voice recorder) ──────────────────────────────────────────────────────

function handleMicClick(): void {
  void voiceStore.toggleRecording();
}

// ── Attach menu ───────────────────────────────────────────────────────────────

function toggleAttachMenu(): void {
  attachMenuOpen.value = !attachMenuOpen.value;
}

function chooseDocument(): void {
  attachMenuOpen.value = false;
  docInputRef.value?.click();
}

function chooseImage(): void {
  attachMenuOpen.value = false;
  imageInputRef.value?.click();
}

/** Shared by both the image and document pickers — addFiles dispatches on type. */
function onFileInputChange(e: Event): void {
  const input = e.target as HTMLInputElement;
  if (input.files?.length) void attachments.addFiles(input.files);
  input.value = ''; // allow re-selecting the same file
}

// ── Thinking-level dropdown ───────────────────────────────────────────────────

function toggleThinkingMenu(): void {
  thinkingMenuOpen.value = !thinkingMenuOpen.value;
}

function selectLevel(next: (typeof THINKING_ITEMS)[number]['level']): void {
  void contextUsage.setLevel(next);
  thinkingMenuOpen.value = false;
}

// ── Outside-click / scroll close (port of app.js + chat_controls.js) ──────────

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
  // Legacy closes the attach menu on body/window scroll (app.js:676-681).
  if (attachMenuOpen.value) attachMenuOpen.value = false;
}

// ── session:turn-interrupted listener ────────────────────────────────────────

function onTurnInterrupted(e: Event): void {
  const detail = (e as CustomEvent<{ text: string }>).detail;
  text.value = detail.text ?? '';
  nextTick(() => {
    grow();
    textareaRef.value?.focus();
    // Move cursor to end (port of app.js _pasteVoiceTranscript).
    const el = textareaRef.value;
    if (el) {
      el.selectionStart = el.selectionEnd = el.value.length;
    }
  });
}

// ── Voice transcript paste (faithful port of app.js _pasteVoiceTranscript) ───

/** Unbind function returned by on('chalie:voice-transcript', …). */
let _unsubVoiceTranscript: (() => void) | null = null;

/**
 * Paste the voice transcript into the compose textarea for review — does NOT
 * auto-send.  Port of app.js _pasteVoiceTranscript (lines 357-368).
 */
function onVoiceTranscript({ text: transcript }: { text: string }): void {
  text.value = transcript;
  nextTick(() => {
    grow();
    textareaRef.value?.focus();
    // Move cursor to end (app.js:367: selectionStart = selectionEnd = text.length).
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

  // Behavioral signals: typing cadence feeds the ambient snapshot.
  if (textareaRef.value) ambient.bindTypingInput(textareaRef.value);

  // Thinking-level: load the persisted override; context indicator: first paint.
  void contextUsage.loadLevel();
  void contextUsage.refresh();

  // Vision-provider gating for the image attach option (app.js:699-706).
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
  voiceStore.destroyRecorder();
});
</script>

<template>
  <footer class="input-dock">
    <!-- Image / document preview strip (renders #imagePreview, self-handles drag-drop). -->
    <ImageAttachStrip />

    <!-- Slide-up attachment menu -->
    <div
      id="attachMenu"
      ref="attachMenuRef"
      class="attach-menu"
      :class="{ hidden: !attachMenuOpen }"
    >
      <div class="attach-menu__inner">
        <button class="attach-menu__item" type="button" @click="chooseDocument">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
            <polyline points="14 2 14 8 20 8"></polyline>
          </svg>
          <span>Attach Document</span>
        </button>
        <button
          v-if="hasVisionProvider"
          id="attachImageBtn"
          class="attach-menu__item"
          type="button"
          @click="chooseImage"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
            <circle cx="8.5" cy="8.5" r="1.5"></circle>
            <polyline points="21 15 16 10 5 21"></polyline>
          </svg>
          <span>Take Photo / Pick Image</span>
        </button>
      </div>
    </div>

    <div class="input-dock__outer">
      <div class="input-dock__inner">
        <!-- + button: opens slide-up attachment menu -->
        <button
          id="attachBtn"
          ref="attachBtnRef"
          class="btn-action btn-action--attach"
          :class="{ active: attachMenuOpen }"
          aria-label="Attach"
          @click.stop="toggleAttachMenu"
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
            <line x1="12" y1="5" x2="12" y2="19"></line>
            <line x1="5" y1="12" x2="19" y2="12"></line>
          </svg>
        </button>

        <!-- Mic button: record voice → transcript posted as a 'voice' turn -->
        <button
          v-if="voiceAvailable"
          id="voiceRecBtn"
          class="btn-icon voice-rec-btn"
          aria-label="Record voice message"
          :data-state="recorderState"
          @click="handleMicClick"
        >
          <svg class="voice-rec-btn__mic" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
            <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
            <line x1="12" y1="19" x2="12" y2="23"></line>
            <line x1="8" y1="23" x2="16" y2="23"></line>
          </svg>
          <span class="voice-rec-btn__dot" aria-hidden="true"></span>
          <span class="voice-rec-btn__spinner" aria-hidden="true"></span>
        </button>

        <!-- Compose textarea -->
        <textarea
          id="chatInput"
          ref="textareaRef"
          v-model="text"
          class="input-dock__textarea"
          placeholder="Talk to Chalie..."
          rows="1"
          @input="grow"
          @keydown="handleKeydown"
        ></textarea>

        <!-- Send / Stop button -->
        <button
          class="btn-action btn-action--send"
          :aria-label="isSending ? 'Stop' : 'Send message'"
          :disabled="!isSending && !canSend"
          @click="isSending ? handleStop() : handleSend()"
        >
          <!-- Send icon -->
          <svg v-if="!isSending" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <line x1="22" y1="2" x2="11" y2="13"></line>
            <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
          </svg>
          <!-- Stop icon (square) -->
          <svg v-else width="16" height="16" viewBox="0 0 24 24" fill="currentColor" stroke="none">
            <rect x="3" y="3" width="18" height="18" rx="2"></rect>
          </svg>
        </button>
      </div>
    </div>

    <!-- Composer controls: thinking-level override (left) + context-size indicator (right) -->
    <div class="dock-controls">
      <div ref="thinkingWrapRef" class="thinking-select">
        <button
          id="thinkingTrigger"
          class="thinking-select__trigger"
          type="button"
          aria-haspopup="true"
          :aria-expanded="thinkingMenuOpen"
          @click.stop="toggleThinkingMenu"
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

    <!-- Mic error label — shown below the dock when mic access / STT fails -->
    <p v-if="recError" id="voiceRecError" class="voice-rec-error">{{ recError }}</p>

    <!-- Hidden image picker. No capture attr: lets mobile show the standard OS
         picker (photo library + take-photo), WhatsApp-style. -->
    <input
      id="imageFileInput"
      ref="imageInputRef"
      type="file"
      accept="image/jpeg,image/png,image/webp,image/gif"
      hidden
      @change="onFileInputChange"
    />

    <!-- Hidden document picker — accepts the legacy document set (index.html:319). -->
    <input
      ref="docInputRef"
      type="file"
      accept=".pdf,.docx,.pptx,.html,.htm,.txt,.md,.csv,.json,.xml"
      hidden
      @change="onFileInputChange"
    />
  </footer>
</template>
