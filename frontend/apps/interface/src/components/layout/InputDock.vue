<script setup lang="ts">
/**
 * InputDock — compose / send / stop.
 *
 * Port sources:
 *   legacy app.js _initInput (auto-grow, send-on-click, onSendComplete focus)
 *   legacy chat.js sendMessage / requestStop (orchestration is in session store)
 *   legacy index.html .input-dock markup (classes preserved for interface.scss)
 *   legacy style.css §10–12 (already in interface.scss; no new global CSS added here)
 *
 * Intentionally unimplemented (wired in P1c):
 *   - Mic recording / voice source
 *   - Image/document attach handlers
 *   - Context-usage bar values
 *   - Thinking-level dropdown logic
 */
import { ref, computed, onMounted, onBeforeUnmount, nextTick } from 'vue';
import { storeToRefs } from 'pinia';
import { useSessionStore } from '../../stores/session';
import { useVoiceStore } from '../../stores/voice';

// ── Stores ────────────────────────────────────────────────────────────────────

const session = useSessionStore();
const voiceStore = useVoiceStore();
const { isSending } = storeToRefs(session);
const { available: voiceAvailable } = storeToRefs(voiceStore);

// ── Local state ───────────────────────────────────────────────────────────────

const textareaRef = ref<HTMLTextAreaElement | null>(null);
const text = ref('');

/** True while a send request is active (mirrors session.isSending reactively). */
const canSend = computed(() => text.value.trim().length > 0);

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
  // session.sendMessage no-ops on empty text, but guard here too so we don't
  // clear the textarea or re-grow unnecessarily.
  if (!trimmed) return;

  // Clear textarea before awaiting so UI feels instant.
  text.value = '';
  await nextTick();
  grow();

  await session.sendMessage(trimmed);
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
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    if (isSending.value) {
      // Mid-ACT: pressing Enter appends to the current turn via sendMessage.
      handleSend();
    } else {
      handleSend();
    }
  }
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

onMounted(() => {
  document.addEventListener('session:turn-interrupted', onTurnInterrupted);
});

onBeforeUnmount(() => {
  document.removeEventListener('session:turn-interrupted', onTurnInterrupted);
});
</script>

<template>
  <footer class="input-dock">
    <!-- Image preview strip — P1c wires real images; slot is structurally present -->
    <!-- <div class="image-preview hidden"></div> -->

    <!-- Attach menu — P1c wires handlers; structurally present but inert -->
    <!-- <div class="attach-menu hidden"> ... </div> -->

    <div class="input-dock__outer">
      <div class="input-dock__inner">
        <!-- + attach button — inert until P1c -->
        <button
          class="btn-action btn-action--attach"
          aria-label="Attach"
          disabled
          title="Attach (coming soon)"
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
            <line x1="12" y1="5" x2="12" y2="19"></line>
            <line x1="5" y1="12" x2="19" y2="12"></line>
          </svg>
        </button>

        <!-- Mic button — visible only when voice is available (gated by P1c for real handling) -->
        <button
          v-if="voiceAvailable"
          class="btn-icon voice-rec-btn"
          aria-label="Record voice message"
          data-state="idle"
          disabled
          title="Voice recording (coming soon)"
        >
          <svg class="voice-rec-btn__mic" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
            <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
            <line x1="12" y1="19" x2="12" y2="23"></line>
            <line x1="8" y1="23" x2="16" y2="23"></line>
          </svg>
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

    <!-- Composer controls: thinking-level (left) + context-size (right) — values wired in C8 -->
    <div class="dock-controls">
      <div class="thinking-select">
        <button
          class="thinking-select__trigger"
          type="button"
          aria-haspopup="true"
          aria-expanded="false"
          disabled
        >
          <span class="thinking-select__caption">Thinking</span>
          <span class="thinking-select__value">Auto</span>
        </button>
      </div>
      <div class="context-display">
        <span class="context-display__caption">Context</span>
        <span class="context-indicator" title="Last request size / context window">—</span>
      </div>
    </div>

    <!-- Hidden file input for image attachment (P1c wires the picker) -->
    <input
      type="file"
      id="imageFileInput"
      accept="image/jpeg,image/png,image/webp,image/gif"
      hidden
    />
  </footer>
</template>

<style scoped lang="scss">
// Mic button disabled state — no visible change needed beyond the disabled attribute,
// but ensure the cursor communicates the state.
.voice-rec-btn:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}

// Attach button disabled state.
.btn-action--attach:disabled {
  opacity: 0.35;
  cursor: not-allowed;
}

// Thinking-select trigger disabled state.
.thinking-select__trigger:disabled {
  cursor: default;
  opacity: 0.65;
}
</style>
