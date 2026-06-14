<script setup lang="ts">
/**
 * DocumentConfirmCard — synthesis card + duplicate-replace prompt.
 *
 * Port of frontend/interface/document_upload.js _showDocumentSynthesis.
 *
 * Rendered when attachments.docCard is non-null.
 * Also renders the duplicate Replace/Keep-Both prompt when attachments.duplicate
 * is set (from uploadDocument finding an existing matching file).
 *
 * Synthesis text: rendered as plain text via {{ }} — the legacy used escHtml()
 * which HTML-escapes the string, then set it as innerHTML.  That is equivalent
 * to binding as text content (no markup is ever rendered).  This matches the
 * legacy intent and is safer (no XSS via document names / LLM text).
 *
 * Key facts: each fact is a plain-text string rendered in a <span>; same logic.
 */
import { ref } from 'vue';
import { useAttachmentsStore } from '../../stores/attachments';
import { showToast } from '../../utils/toast';

const attachments = useAttachmentsStore();

// ── Augment-area toggle (mirrors legacy augmentBtn click handler) ─────────────
const showAugmentArea = ref(false);
const augmentText = ref('');

// ── Loading state per action ──────────────────────────────────────────────────
const confirmPending = ref(false);
const augmentPending = ref(false);
const discardPending = ref(false);

function resetActionState(): void {
  confirmPending.value = false;
  augmentPending.value = false;
  discardPending.value = false;
  showAugmentArea.value = false;
  augmentText.value = '';
}

async function onConfirm(): Promise<void> {
  if (!attachments.docCard) return;
  confirmPending.value = true;
  try {
    await attachments.confirmDoc(attachments.docCard.id);
    resetActionState();
  } catch {
    confirmPending.value = false;
    showToast('Confirmation failed');
  }
}

async function onAugmentSubmit(): Promise<void> {
  const ctx = augmentText.value.trim();
  if (!ctx || !attachments.docCard) return;
  augmentPending.value = true;
  try {
    await attachments.augmentDoc(attachments.docCard.id, ctx);
    resetActionState();
  } catch {
    augmentPending.value = false;
    showToast('Failed to save context');
  }
}

async function onDiscard(): Promise<void> {
  if (!attachments.docCard) return;
  discardPending.value = true;
  try {
    await attachments.discardDoc(attachments.docCard.id);
    resetActionState();
  } catch {
    discardPending.value = false;
    showToast('Discard failed');
  }
}

// ── Duplicate actions ─────────────────────────────────────────────────────────

async function onReplace(): Promise<void> {
  if (!attachments.duplicate) return;
  const { newId, dup } = attachments.duplicate;
  await attachments.replaceDuplicate(newId, dup.id);
}

async function onKeepBoth(): Promise<void> {
  if (!attachments.duplicate) return;
  await attachments.keepBothDuplicate(attachments.duplicate.newId);
}

function formatDate(isoString: string | null): string {
  if (!isoString) return '';
  return new Date(isoString).toLocaleDateString();
}
</script>

<template>
  <!-- Duplicate-replace prompt -->
  <div
    v-if="attachments.duplicate"
    id="uploadDuplicateWarning"
    class="upload-duplicate"
  >
    <div>
      This looks like an updated version of
      <strong>{{ attachments.duplicate.dup.original_name }}</strong>
      <template v-if="attachments.duplicate.dup.created_at">
        from {{ formatDate(attachments.duplicate.dup.created_at) }}
      </template>
      . Replace the older version, or keep both?
    </div>
    <div class="upload-duplicate__actions">
      <button
        class="upload-duplicate__btn upload-duplicate__btn--primary"
        type="button"
        @click="onReplace"
      >
        Replace
      </button>
      <button
        class="upload-duplicate__btn"
        type="button"
        @click="onKeepBoth"
      >
        Keep Both
      </button>
    </div>
  </div>

  <!-- Synthesis card -->
  <div
    v-if="attachments.docCard"
    class="doc-synthesis-card"
    :data-doc-id="attachments.docCard.id"
  >
    <div class="doc-synthesis__header">
      <span class="doc-synthesis__name">{{ attachments.docCard.name }}</span>
      <span
        v-if="attachments.docCard.typeLabel && attachments.docCard.typeLabel !== 'document'"
        class="doc-synthesis__tag"
      >
        {{ attachments.docCard.typeLabel }}
      </span>
    </div>

    <!-- Synthesis text: plain text, matching legacy escHtml usage (no markup). -->
    <p class="doc-synthesis__text">{{ attachments.docCard.synthesis }}</p>

    <!-- Key facts -->
    <div v-if="attachments.docCard.keyFacts.length" class="doc-synthesis__facts">
      <span
        v-for="(fact, i) in attachments.docCard.keyFacts"
        :key="i"
        class="doc-synthesis__fact"
      >
        {{ fact }}
      </span>
    </div>

    <!-- Action buttons -->
    <div class="doc-synthesis__actions">
      <button
        class="doc-synthesis__btn doc-synthesis__btn--confirm"
        type="button"
        :disabled="confirmPending"
        @click="onConfirm"
      >
        {{ confirmPending ? 'Confirming…' : 'Looks good' }}
      </button>

      <button
        class="doc-synthesis__btn doc-synthesis__btn--augment"
        type="button"
        @click="showAugmentArea = !showAugmentArea"
      >
        Add context
      </button>

      <button
        class="doc-synthesis__btn doc-synthesis__btn--discard"
        type="button"
        :disabled="discardPending"
        @click="onDiscard"
      >
        {{ discardPending ? 'Discarding…' : 'Discard' }}
      </button>
    </div>

    <!-- Augment area (toggled by "Add context" button) -->
    <div
      v-show="showAugmentArea"
      class="doc-synthesis__augment-area"
    >
      <textarea
        v-model="augmentText"
        class="doc-synthesis__textarea"
        placeholder="Add context about this document..."
      />
      <button
        class="doc-synthesis__btn doc-synthesis__btn--submit"
        type="button"
        :disabled="augmentPending"
        @click="onAugmentSubmit"
      >
        {{ augmentPending ? 'Saving…' : 'Save' }}
      </button>
    </div>
  </div>
</template>

<style scoped lang="scss">
// ── Duplicate warning ─────────────────────────────────────────────────────────
.upload-duplicate {
  padding: var(--space-sm, 10px) var(--space-md, 16px);
  background: var(--surface-raised, var(--surface, #1a1a1a));
  border: 1px solid var(--border, rgba(255 255 255 / 0.12));
  border-radius: var(--radius, 8px);
  font-size: 0.875rem;
  color: var(--text, currentColor);
  display: flex;
  flex-direction: column;
  gap: var(--space-sm, 10px);

  &__actions {
    display: flex;
    gap: var(--space-xs, 6px);
  }

  &__btn {
    padding: 5px 14px;
    border-radius: var(--radius-sm, 6px);
    border: 1px solid var(--border, rgba(255 255 255 / 0.2));
    background: transparent;
    color: var(--text, currentColor);
    font-size: 0.8125rem;
    cursor: pointer;
    transition: background 0.15s;

    &:hover {
      background: var(--surface-hover, rgba(255 255 255 / 0.06));
    }

    &--primary {
      background: var(--accent, #6366f1);
      border-color: var(--accent, #6366f1);
      color: #fff;

      &:hover {
        background: var(--accent-hover, #4f46e5);
        border-color: var(--accent-hover, #4f46e5);
      }
    }
  }
}

// ── Synthesis card ────────────────────────────────────────────────────────────
.doc-synthesis-card {
  padding: var(--space-md, 16px);
  background: var(--surface-raised, var(--surface, #1a1a1a));
  border: 1px solid var(--border, rgba(255 255 255 / 0.12));
  border-radius: var(--radius, 8px);
  display: flex;
  flex-direction: column;
  gap: var(--space-sm, 10px);
}

.doc-synthesis {
  &__header {
    display: flex;
    align-items: baseline;
    gap: var(--space-xs, 6px);
    flex-wrap: wrap;
  }

  &__name {
    font-weight: 600;
    font-size: 0.9375rem;
    color: var(--text, currentColor);
  }

  &__tag {
    font-size: 0.75rem;
    padding: 2px 8px;
    border-radius: 12px;
    background: var(--accent-muted, rgba(99 102 241 / 0.18));
    color: var(--accent, #6366f1);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-weight: 500;
  }

  &__text {
    margin: 0;
    font-size: 0.875rem;
    line-height: 1.55;
    color: var(--text-secondary, var(--text, currentColor));
    white-space: pre-wrap;
  }

  &__facts {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-xs, 6px);
  }

  &__fact {
    font-size: 0.8125rem;
    padding: 3px 10px;
    border-radius: 12px;
    background: var(--surface-overlay, rgba(255 255 255 / 0.06));
    color: var(--text-secondary, var(--text, currentColor));
    border: 1px solid var(--border, rgba(255 255 255 / 0.1));
  }

  &__actions {
    display: flex;
    gap: var(--space-xs, 6px);
    flex-wrap: wrap;
  }

  &__btn {
    padding: 5px 14px;
    border-radius: var(--radius-sm, 6px);
    border: 1px solid var(--border, rgba(255 255 255 / 0.2));
    background: transparent;
    color: var(--text, currentColor);
    font-size: 0.8125rem;
    cursor: pointer;
    transition: background 0.15s;

    &:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }

    &:not(:disabled):hover {
      background: var(--surface-hover, rgba(255 255 255 / 0.06));
    }

    &--confirm {
      background: var(--accent, #6366f1);
      border-color: var(--accent, #6366f1);
      color: #fff;

      &:not(:disabled):hover {
        background: var(--accent-hover, #4f46e5);
        border-color: var(--accent-hover, #4f46e5);
      }
    }

    &--discard {
      color: var(--error, #f87171);
      border-color: var(--error-muted, rgba(248 113 113 / 0.3));

      &:not(:disabled):hover {
        background: var(--error-surface, rgba(248 113 113 / 0.08));
      }
    }
  }

  &__augment-area {
    display: flex;
    flex-direction: column;
    gap: var(--space-xs, 6px);
  }

  &__textarea {
    width: 100%;
    min-height: 72px;
    padding: var(--space-xs, 8px);
    border-radius: var(--radius-sm, 6px);
    border: 1px solid var(--border, rgba(255 255 255 / 0.15));
    background: var(--input-bg, var(--surface, #111));
    color: var(--text, currentColor);
    font-size: 0.875rem;
    font-family: inherit;
    resize: vertical;
    box-sizing: border-box;

    &::placeholder {
      color: var(--text-tertiary, rgba(255 255 255 / 0.35));
    }

    &:focus {
      outline: none;
      border-color: var(--accent, #6366f1);
    }
  }
}
</style>
