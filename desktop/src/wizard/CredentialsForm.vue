<script setup lang="ts">
import { ref } from 'vue';

import { connect } from './api';
import { describeError, isKind } from './errors';
import Message from './Message.vue';

const props = defineProps<{
  host: string;
  port: number;
  username: string;
  notice?: string;
}>();

const emit = defineEmits<{ (e: 'back'): void }>();

const username = ref(props.username);
const password = ref('');
const busy = ref(false);
const failure = ref<{ text: string; detail: string } | null>(null);

// A successful connect navigates this window to the server, so the form stays busy until
// the page it lives on is replaced; only a failure ever comes back here.
async function submit(): Promise<void> {
  busy.value = true;
  failure.value = null;
  try {
    await connect(props.host, props.port, username.value.trim(), password.value);
  } catch (error) {
    failure.value = describeError(error);
    if (isKind(error, 'invalid_credentials')) password.value = '';
    busy.value = false;
  }
}
</script>

<template>
  <h1>Sign in</h1>
  <p class="lead">Your Chalie login for {{ host }}:{{ port }}.</p>

  <form @submit.prevent="submit">
    <fieldset :disabled="busy">
      <Message v-if="notice" :text="notice" />

      <label for="username">Username</label>
      <input id="username" v-model="username" required autocapitalize="off" autocorrect="off"
             spellcheck="false" autocomplete="username" />

      <label for="password">Password</label>
      <input id="password" v-model="password" type="password" required
             autocomplete="current-password" />

      <Message v-if="failure" tone="error" :text="failure.text" :detail="failure.detail" />

      <div class="actions">
        <button type="submit" class="primary">{{ busy ? 'Connecting…' : 'Connect' }}</button>
        <button type="button" @click="emit('back')">Back</button>
      </div>
    </fieldset>
  </form>
</template>
