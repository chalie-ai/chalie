<script setup lang="ts">
// A Chalie is already running and answering, but has nobody to sign in as. Nothing is
// installed or started here — only the account it is missing is made.
import { computed, ref } from 'vue';

import { createAccount } from './api';
import { describeError } from './errors';
import AccountForm from './AccountForm.vue';

const props = defineProps<{ host: string; port: number; notice?: string }>();

const emit = defineEmits<{ (e: 'back'): void }>();

const lead = computed(
  () =>
    `The first account on ${props.host}:${props.port}. ` +
    'This app remembers it so you are signed in on every launch.',
);

const busy = ref(false);
const failure = ref<{ text: string; detail: string } | null>(null);

// Only a failure ever comes back here: the account being made is also what signs this window
// in, so success navigates the window to that Chalie and replaces this page.
async function submit(username: string, password: string): Promise<void> {
  busy.value = true;
  failure.value = null;
  try {
    await createAccount(props.host, props.port, username, password);
  } catch (error) {
    failure.value = describeError(error);
    busy.value = false;
  }
}
</script>

<template>
  <AccountForm
    heading="Create your account"
    :lead="lead"
    :notice="notice"
    submit-label="Create account"
    :busy="busy"
    :failure="failure"
    @submit="submit"
    @back="emit('back')"
  />
</template>
