<script setup lang="ts">
// The login for a Chalie that has nobody to sign in as yet. Both the way in — a Chalie
// being installed here, and one already running with no account — ask for exactly this.
import { computed, ref } from 'vue';

import Message from './Message.vue';

// What the server accepts, mirrored here so a refusal it is certain to give costs nobody a
// round trip. Nothing stricter than the server's own rule: a rule invented here would turn
// away passwords Chalie is perfectly happy with.
const MIN_PASSWORD_LENGTH = 8;

const props = defineProps<{
  heading: string;
  lead: string;
  notice?: string;
  submitLabel: string;
  busy: boolean;
  failure?: { text: string; detail: string } | null;
  /** What to start the username field with, so details a server turned down come back to be
   * corrected rather than typed again. */
  username?: string;
}>();

const emit = defineEmits<{
  (e: 'submit', username: string, password: string): void;
  (e: 'back'): void;
}>();

const username = ref(props.username ?? '');
const password = ref('');
const confirmation = ref('');
// Nothing is said about details that have not been filled in yet: the complaint appears when
// somebody first asks to go on, and from then on it keeps up with what they type.
const asked = ref(false);

const problem = computed(() => {
  if (username.value.trim() === '') return 'Choose a username.';
  if (password.value.length < MIN_PASSWORD_LENGTH) {
    return `The password needs at least ${MIN_PASSWORD_LENGTH} characters.`;
  }
  if (confirmation.value !== password.value) return 'The two passwords do not match.';
  return '';
});

function submit(): void {
  asked.value = true;
  if (problem.value !== '') return;
  emit('submit', username.value.trim(), password.value);
}
</script>

<template>
  <h1>{{ heading }}</h1>
  <p class="lead">{{ lead }}</p>

  <form @submit.prevent="submit">
    <fieldset :disabled="busy">
      <Message v-if="notice" :text="notice" />

      <label for="username">Username</label>
      <input id="username" v-model="username" autocapitalize="off" autocorrect="off"
             spellcheck="false" autocomplete="username" />

      <label for="password">Password</label>
      <input id="password" v-model="password" type="password" autocomplete="new-password" />

      <label for="confirmation">Password again</label>
      <input id="confirmation" v-model="confirmation" type="password"
             autocomplete="new-password" />

      <Message v-if="asked && problem" tone="error" :text="problem" />
      <Message v-else-if="failure" tone="error" :text="failure.text" :detail="failure.detail" />

      <div class="actions">
        <button type="submit" class="primary">{{ busy ? 'Working…' : submitLabel }}</button>
        <button type="button" @click="emit('back')">Back</button>
      </div>
    </fieldset>
  </form>
</template>
