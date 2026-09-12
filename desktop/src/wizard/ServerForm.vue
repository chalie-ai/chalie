<script setup lang="ts">
import { ref, watch } from 'vue';

import { DEFAULT_PORT, probeServer } from './api';
import type { ServerStatus } from './api';
import { describeError } from './errors';
import Message from './Message.vue';

const emit = defineEmits<{
  /** `hasAccount` is false for a Chalie nobody has made an account on yet: there is nobody
   * to sign in as, so making that account is what comes next. */
  (e: 'ready', host: string, port: number, hasAccount: boolean): void;
  (e: 'back'): void;
}>();

const host = ref('');
const port = ref(DEFAULT_PORT);
const busy = ref(false);
const status = ref<ServerStatus | null>(null);
const failure = ref<{ text: string; detail: string } | null>(null);

// A verdict is about the address it was reached for. Once either half of that address
// changes it says nothing about where Continue would go, so the new one is checked first.
watch([host, port], () => {
  status.value = null;
  failure.value = null;
});

async function check(): Promise<void> {
  busy.value = true;
  status.value = null;
  failure.value = null;
  try {
    status.value = await probeServer(host.value.trim(), port.value);
  } catch (error) {
    failure.value = describeError(error);
  } finally {
    busy.value = false;
  }
}
</script>

<template>
  <h1>Where is Chalie?</h1>
  <p class="lead">Enter the address of the Chalie server you want to use.</p>

  <form @submit.prevent="check">
    <fieldset :disabled="busy">
      <label for="host">Host</label>
      <input id="host" v-model="host" required autocapitalize="off" autocorrect="off"
             spellcheck="false" placeholder="chalie.example.com" />

      <label for="port">Port</label>
      <input id="port" v-model.number="port" type="number" required min="1" max="65535" />

      <Message v-if="failure" tone="error" :text="failure.text" :detail="failure.detail" />
      <Message
        v-else-if="status && !status.has_master_account"
        text="Chalie answered. It has no account yet, so you will create one next."
      />
      <Message
        v-else-if="status"
        :text="`Chalie answered. Its vault is ${status.vault_state}.`"
      />

      <div class="actions">
        <button v-if="status" type="button" class="primary"
                @click="emit('ready', host.trim(), port, status.has_master_account)">
          Continue
        </button>
        <button v-else type="submit" class="primary">
          {{ busy ? 'Checking…' : 'Check' }}
        </button>
        <button type="button" @click="emit('back')">Back</button>
      </div>
    </fieldset>
  </form>
</template>
