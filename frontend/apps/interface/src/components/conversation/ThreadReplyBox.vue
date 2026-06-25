<script setup lang="ts">
/**
 * ThreadReplyBox — a compact compose box rendered at the bottom of an expanded
 * thread. Sends a reply that appends to the existing thread (turn_id) instead of
 * opening a new one.
 */
import { ref, computed, nextTick } from 'vue';
import { Send } from '@lucide/vue';
import { useSessionStore } from '../../stores/session';

const props = defineProps<{ turnId: number }>();

const session = useSessionStore();
const textareaRef = ref<HTMLTextAreaElement | null>(null);
const text = ref('');

const canSend = computed(() => text.value.trim().length > 0);

function grow(): void {
  const el = textareaRef.value;
  if (!el) return;
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}

async function handleSend(): Promise<void> {
  const trimmed = text.value.trim();
  if (!trimmed) return;
  text.value = '';
  await nextTick();
  grow();
  await session.sendMessage(trimmed, 'text', [], [], props.turnId);
  textareaRef.value?.focus();
}
</script>

<template>
  <div class="thread-reply">
    <textarea
      ref="textareaRef"
      v-model="text"
      class="thread-reply__textarea"
      placeholder="Reply in thread..."
      rows="1"
      @input="grow"
      @keydown.enter.exact.prevent="handleSend"
    ></textarea>
    <button
      class="btn-action btn-action--send thread-reply__send"
      aria-label="Send reply"
      :disabled="!canSend"
      @click="handleSend()"
    >
      <Send :size="18" />
    </button>
  </div>
</template>

<style scoped lang="scss">
.thread-reply {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
  background: var(--bg-surface-2);
  border: 1px solid var(--border);
  border-radius: var(--radius-full);
  padding: var(--space-xs) var(--space-md);
  margin: var(--space-sm) 0 var(--space-md);
  backdrop-filter: blur(20px) saturate(120%);
  -webkit-backdrop-filter: blur(20px) saturate(120%);
  box-shadow: var(--shadow-card);
  transition: border-color var(--duration-fast) ease;
  width: 90%;
  align-self: flex-end;
}

.thread-reply:focus-within {
  border-color: color-mix(in oklab, var(--violet) 35%, transparent);
}

.thread-reply__textarea {
  flex: 1;
  background: transparent;
  border: none;
  outline: none;
  color: var(--text-primary);
  font-size: 0.875rem;
  line-height: 1.5;
  resize: none;
  max-height: 120px;
  padding: var(--space-xs) 0;
}

.thread-reply__textarea::placeholder {
  color: var(--text-tertiary);
}

.thread-reply__send {
  flex-shrink: 0;
}
</style>