<script setup lang="ts">
import { computed, onMounted, ref } from 'vue';

import {
  DEFAULT_PORT,
  autoConnect,
  cancelInstall,
  detectLocal,
  getSetupState,
  onInstallPhase,
} from './wizard/api';
import type { InstallPhase, UnlistenFn } from './wizard/api';
import { describeError } from './wizard/errors';
import CreateAccount from './wizard/CreateAccount.vue';
import CredentialsForm from './wizard/CredentialsForm.vue';
import InstallLocal from './wizard/InstallLocal.vue';
import Message from './wizard/Message.vue';
import ModeChoice from './wizard/ModeChoice.vue';
import ServerForm from './wizard/ServerForm.vue';

type Step =
  | 'loading'
  | 'connecting'
  | 'mode'
  | 'detecting'
  | 'server'
  | 'credentials'
  | 'account'
  | 'install';

// Why the sign-in form is being shown for a Chalie on this Mac. Both of the ways there are
// the same discovery — there is one here already, with an account on it — arrived at from
// two different questions.
const ALREADY_RUNNING = 'Chalie is already running on this Mac.';
const ALREADY_INSTALLED =
  'Chalie is already installed and running on this Mac, so there is nothing to install. ' +
  'Sign in to it.';

// Why the form that makes an account is being shown. A Chalie found on this Mac and one at
// an address somebody typed can both turn out the same way — answering, with nobody to sign
// in as — and are told so in the same words.
const NO_ACCOUNT_YET = 'This Chalie has no account yet. Create one.';

// Shown on the "connecting" step only for the two phases a native start ever reaches: this
// Mac's own Chalie is started the same way an install starts one, so its own wording covers
// what is happening instead of leaving "Connecting…" up while nothing is answering yet.
const NATIVE_START_TEXT: Partial<Record<InstallPhase, string>> = {
  starting: 'Starting Chalie on this Mac…',
  waiting: 'Waiting for Chalie to answer…',
};

const step = ref<Step>('loading');
const host = ref('');
const port = ref(DEFAULT_PORT);
const username = ref('');
const notice = ref('');
const failure = ref<{ text: string; detail: string } | null>(null);
const nativeStartPhase = ref<InstallPhase | null>(null);

const connectingLead = computed(
  () =>
    (nativeStartPhase.value && NATIVE_START_TEXT[nativeStartPhase.value]) ||
    `Connecting to ${host.value}:${port.value}…`,
);

// A step can be left before the command it is waiting on answers — every Back and Cancel
// does exactly that. The token makes the late answer land on the floor instead of
// dragging the user back to the step they walked away from.
let current = 0;

function begin(next: Step): number {
  current += 1;
  step.value = next;
  failure.value = null;
  return current;
}

onMounted(start);

async function start(): Promise<void> {
  const token = begin('loading');
  try {
    const setup = await getSetupState();
    if (token !== current) return;
    if (setup.host == null || setup.port == null) {
      step.value = 'mode';
      return;
    }
    host.value = setup.host;
    port.value = setup.port;
    await resume();
  } catch (error) {
    if (token !== current) return;
    failure.value = describeError(error);
    step.value = 'mode';
  }
}

// Silent relaunch login. When it works the window is navigated to the server and this page
// is replaced, so "Connecting…" is deliberately left on screen.
//
// This is also where a native start shows itself: the same phase events an install streams
// arrive here whenever the Rust side starts this Mac's own Chalie instead of just reporting
// it unreachable, so the listener is registered around the one call that can trigger it — a
// plain function has no mount/unmount of its own to hang it on, so it comes down again in
// `finally` instead.
async function resume(): Promise<void> {
  const token = begin('connecting');
  nativeStartPhase.value = null;
  let unlisten: UnlistenFn | undefined;
  try {
    unlisten = await onInstallPhase((phase) => {
      if (token === current) nativeStartPhase.value = phase;
    });
    const result = await autoConnect();
    if (token !== current || !result.needs_credentials) return;
    // A username comes back only when one was stored and the server turned it down.
    // Without one the settings file held no login to refuse, which is a different
    // thing to tell somebody.
    username.value = result.username ?? '';
    notice.value =
      result.username == null
        ? 'No saved login was found for this Chalie. Sign in once and it will be remembered.'
        : 'The saved Chalie login was refused. Sign in again.';
    step.value = 'credentials';
  } catch (error) {
    if (token !== current) return;
    // A failed start is not still waiting on anything, so the lead line stops claiming it is.
    nativeStartPhase.value = null;
    failure.value = describeError(error, 'start');
  } finally {
    void unlisten?.();
  }
}

async function chooseMode(choice: 'existing' | 'install'): Promise<void> {
  const token = begin('detecting');
  try {
    const local = await detectLocal();
    if (token !== current) return;
    // The address a local Chalie answers at, which is the same one an install here puts it
    // on, so it is kept whether or not anything is answering there yet.
    host.value = local.host;
    port.value = local.port;
    if (local.found) {
      // A Chalie that is running but has no account yet can only refuse a sign-in. It is
      // one field away from being usable, so it gets the form that makes that account —
      // whichever of the two choices led here.
      if (!local.has_master_account) {
        notice.value = NO_ACCOUNT_YET;
        step.value = 'account';
        return;
      }
      // Asking to install one that is already here is not a mistake to refuse: it is the
      // same wish, already granted, so it is said plainly instead of quietly turning into
      // a different screen than the button promised.
      signInToLocal(
        local.host,
        local.port,
        choice === 'install' ? ALREADY_INSTALLED : ALREADY_RUNNING,
      );
      return;
    }
    step.value = choice === 'install' ? 'install' : 'server';
  } catch (error) {
    if (token !== current) return;
    failure.value = describeError(error);
    step.value = 'mode';
  }
}

// The one way to a Chalie on this Mac that already has an account: the sign-in form, told
// why it is being asked for. An install that finds an account already there arrives here
// too, because there is nothing left for it to install or create.
function signInToLocal(nextHost: string, nextPort: number, why: string): void {
  host.value = nextHost;
  port.value = nextPort;
  username.value = '';
  notice.value = why;
  begin('credentials');
}

function signInInstead(nextHost: string, nextPort: number): void {
  signInToLocal(nextHost, nextPort, ALREADY_INSTALLED);
}

// A Chalie somebody named by its address. One with an account is signed in to; one without
// gets the form that makes it, told why, exactly as one found on this Mac would be.
function useServer(nextHost: string, nextPort: number, hasAccount: boolean): void {
  host.value = nextHost;
  port.value = nextPort;
  username.value = '';
  notice.value = hasAccount ? '' : NO_ACCOUNT_YET;
  begin(hasAccount ? 'credentials' : 'account');
}

// Leaving the connecting step while this Mac's own Chalie is on its way up ends the app's
// wait for it — the boot itself is not the app's to stop, and keeps going in the background.
// Every other way back to the start (Cancel elsewhere, Back from a form) leaves nothing
// running, so nativeStartPhase is unset and this is a plain reset.
async function backToStart(): Promise<void> {
  if (nativeStartPhase.value) {
    nativeStartPhase.value = null;
    try {
      await cancelInstall();
    } catch {
      // The screen is leaving either way.
    }
  }
  username.value = '';
  notice.value = '';
  begin('mode');
}
</script>

<template>
  <main>
    <template v-if="step === 'loading'">
      <h1>Chalie</h1>
      <p class="lead">Reading your setup…</p>
    </template>

    <template v-else-if="step === 'connecting'">
      <h1>Chalie</h1>
      <p class="lead">{{ connectingLead }}</p>
      <Message v-if="failure" tone="error" :text="failure.text" :detail="failure.detail" />
      <div class="actions">
        <button v-if="failure" type="button" class="primary" @click="resume">Try again</button>
        <button type="button" @click="backToStart">
          {{ failure ? 'Use a different Chalie' : 'Cancel' }}
        </button>
      </div>
    </template>

    <template v-else-if="step === 'detecting'">
      <h1>Chalie</h1>
      <p class="lead">Looking for a Chalie on this Mac…</p>
      <div class="actions">
        <button type="button" @click="backToStart">Cancel</button>
      </div>
    </template>

    <ModeChoice v-else-if="step === 'mode'" @choose="chooseMode" />

    <ServerForm v-else-if="step === 'server'" @ready="useServer" @back="backToStart" />

    <CredentialsForm
      v-else-if="step === 'credentials'"
      :key="`${host}:${port}:${username}`"
      :host="host"
      :port="port"
      :username="username"
      :notice="notice"
      @back="backToStart"
    />

    <CreateAccount
      v-else-if="step === 'account'"
      :key="`${host}:${port}`"
      :host="host"
      :port="port"
      :notice="notice"
      @back="backToStart"
    />

    <InstallLocal
      v-else-if="step === 'install'"
      :host="host"
      :port="port"
      @sign-in="signInInstead"
      @back="backToStart"
    />

    <Message
      v-if="failure && step === 'mode'"
      tone="error"
      :text="failure.text"
      :detail="failure.detail"
    />
  </main>
</template>
