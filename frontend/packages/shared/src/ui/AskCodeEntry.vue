<!-- A code ask's answer: one field that takes six ASCII digits and a button
     that sends them as the ask's reply, enabled only once all six are in;
     Enter sends too. The code lives in this field alone and is emptied as it
     is sent. No form element: the row sits inside inline controls, and Enter
     must never reach a form around it. -->
<script setup lang="ts">
import { computed, ref } from 'vue';
import { useAsksStore, type CodeAsk } from '../stores/asks';
import { CODE_ASK_WORDS } from '../utils/askLabels';

const props = defineProps<{ ask: CodeAsk }>();

const asks = useAsksStore();
const code = ref('');
const words = computed(() => CODE_ASK_WORDS[props.ask.data.service]);
const complete = computed(() => /^[0-9]{6}$/.test(code.value));

// Anything but an ASCII digit is dropped as it is typed or pasted; the field
// is rewritten too, since dropping a character can leave `code` unchanged.
function onInput(event: Event): void {
  const field = event.target as HTMLInputElement;
  code.value = field.value.replace(/[^0-9]/g, '').slice(0, 6);
  field.value = code.value;
}

function send(): void {
  if (!complete.value) return;
  const value = code.value;
  code.value = '';
  void asks.reply(props.ask.ask_id, { code: value });
}
</script>

<template>
  <span class="activity__code">
    <label class="activity__code-field">
      <span class="activity__code-label">{{ words.field }}</span>
      <input
        class="activity__code-input"
        type="text"
        inputmode="numeric"
        autocomplete="one-time-code"
        pattern="[0-9]{6}"
        :value="code"
        @input="onInput"
        @keydown.enter.prevent="send"
      />
    </label>
    <button
      type="button"
      class="activity__answer activity__answer--allow"
      :disabled="!complete"
      @click="send"
    >
      {{ words.button }}
    </button>
  </span>
</template>

<style>
/* Not scoped, like AskRow's: the entry sits in the same activity rows in
   both apps. */
.activity__code,
.activity__code-field {
  display: inline-flex;
  flex-shrink: 0;
  align-items: center;
  gap: 8px;
}

.activity__code-label {
  font-family: var(--font-mono);
  font-size: var(--fs-mono);
  color: var(--muted);
  white-space: nowrap;
}

/* Two classes, so it outweighs an app's own full-width text-input rule. */
.activity__code .activity__code-input {
  width: 12ch;
  padding: 0 12px;
  font-family: var(--font-mono);
  font-size: var(--fs-mono);
}

.activity__code > button {
  flex-shrink: 0;
  white-space: nowrap;
}
</style>
