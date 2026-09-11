<script setup lang="ts">
// Putting a Chalie on this Mac, watched while it happens: the account to create first, then
// every line the install prints as it prints it, then either the product UI or the reason
// there is none.
//
// A reason that is about the account details is not a reason to install anything again — by
// the time an account can be refused, Chalie is installed, started and answering. Details it
// turned down come back to this screen's form to be corrected; an account that is already
// there cannot be created at all, so that one leaves for the sign-in form instead.
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue';

import {
  cancelInstall,
  createAccount,
  installLocal,
  onInstallOutput,
  onInstallPhase,
} from './api';
import type { InstallOutput, InstallPhase, UnlistenFn } from './api';
import { describeError, isKind } from './errors';
import AccountForm from './AccountForm.vue';
import Message from './Message.vue';

const props = defineProps<{ host: string; port: number }>();

const emit = defineEmits<{
  (e: 'back'): void;
  /** Nothing is left to install here, and nothing to create: this Chalie has an account
   * already, so somebody has to sign in to it instead. */
  (e: 'sign-in', host: string, port: number): void;
}>();

// How much of the output stays on screen. Enough to scroll back through what the installer
// did, bounded so a command that prints megabytes cannot grow the window without end.
const KEPT_LINES = 400;

const LEAD =
  'Chalie needs an account. This login is the one you will use for the Chalie being ' +
  'installed here, and this app remembers it so you are signed in on every launch.';

// The lead the second time the form is asked for. Without it, correcting a password looks
// like starting the whole install over, which is the one thing it is not.
const ACCOUNT_LEAD =
  'Chalie is installed and running on this Mac. Only the account details were refused, so ' +
  'correcting them here creates the account — nothing is installed again.';

const PHASES: Record<InstallPhase, string> = {
  installing: 'Installing Chalie…',
  starting: 'Starting Chalie…',
  waiting: 'Waiting for Chalie to answer…',
  creating_account: 'Creating your account…',
  connecting: 'Opening Chalie…',
};

// The phases a Cancel still has something to stop. After them the install is one request
// and a navigation away, with no command left running, so the button goes rather than
// staying on screen doing nothing.
const STOPPABLE: InstallPhase[] = ['installing', 'starting', 'waiting'];

interface Line {
  id: number;
  stream: InstallOutput['stream'];
  text: string;
}

// Which of this screen's three faces is showing, and there is only ever one: the account to
// create, the install itself, or that same account again once Chalie has had its say on it.
type Face = 'details' | 'install' | 'account';

const face = ref<Face>('details');
const login = ref<{ username: string; password: string } | null>(null);
const running = ref(false);
const busy = ref(false);
const phase = ref<InstallPhase>('installing');
const lines = ref<Line[]>([]);
// The failure itself rather than the sentence for it, because what it was decides where it
// leaves somebody as well as what they are told.
const problem = ref<unknown>(null);
const panel = ref<HTMLElement | null>(null);

const failure = computed(() => (problem.value == null ? null : describeError(problem.value)));
const hasAccountAlready = computed(() => isKind(problem.value, 'account_exists'));

let nextLine = 0;
// Kept as promises rather than awaited here: a component torn down before the listeners are
// registered would otherwise leave them registered for good.
const listeners: Promise<UnlistenFn>[] = [];

onMounted(() => {
  listeners.push(
    onInstallPhase((next) => {
      phase.value = next;
    }),
    onInstallOutput((output) => {
      lines.value.push({ id: (nextLine += 1), stream: output.stream, text: output.line });
      if (lines.value.length > KEPT_LINES) {
        lines.value.splice(0, lines.value.length - KEPT_LINES);
      }
    }),
  );
});

onUnmounted(() => {
  for (const listener of listeners) void listener.then((stop) => stop());
});

// The newest line is the one being waited on, so the panel follows it rather than leaving
// somebody to drag a scrollbar through an install.
watch(
  () => lines.value.length,
  () => {
    void nextTick(() => {
      if (panel.value) panel.value.scrollTop = panel.value.scrollHeight;
    });
  },
);

// Where a failure leaves somebody. Details Chalie turned down are corrected on the form;
// everything else is read on the install's own screen, with what it printed still above it —
// an account that already exists included, because no amount of correcting makes one of
// those creatable.
function landsOn(error: unknown): Face {
  return isKind(error, 'invalid_account') ? 'account' : 'install';
}

function begin(username: string, password: string): void {
  login.value = { username, password };
  face.value = 'install';
  void run();
}

// Only a failure ever comes back here: a finished install hands this window its session and
// navigates it to the Chalie that was just installed, replacing this page.
async function run(): Promise<void> {
  const details = login.value;
  if (!details) return;
  running.value = true;
  problem.value = null;
  lines.value = [];
  phase.value = 'installing';
  try {
    await installLocal(details.username, details.password);
  } catch (error) {
    problem.value = error;
    running.value = false;
    face.value = landsOn(error);
  }
}

// The corrected details, given to the Chalie that is already here. Nothing is installed or
// started a second time — this is the same request the install made on its own at the end,
// with what was refused put right.
async function makeAccount(username: string, password: string): Promise<void> {
  login.value = { username, password };
  busy.value = true;
  problem.value = null;
  try {
    await createAccount(props.host, props.port, username, password);
  } catch (error) {
    problem.value = error;
    busy.value = false;
    face.value = landsOn(error);
  }
}

// Stopping does not end the wait here — the install itself reports the cancellation, which
// is what puts this screen into its stopped state, so there is one account of what happened.
async function stop(): Promise<void> {
  try {
    await cancelInstall();
  } catch (error) {
    problem.value = error;
  }
}
</script>

<template>
  <AccountForm
    v-if="face === 'details'"
    heading="Install Chalie on this Mac"
    :lead="LEAD"
    submit-label="Install"
    :busy="false"
    @submit="begin"
    @back="emit('back')"
  />

  <AccountForm
    v-else-if="face === 'account'"
    heading="Create your account"
    :lead="ACCOUNT_LEAD"
    :username="login?.username"
    submit-label="Create account"
    :busy="busy"
    :failure="failure"
    @submit="makeAccount"
    @back="emit('back')"
  />

  <template v-else>
    <h1>Installing Chalie</h1>
    <p class="lead">{{ running ? PHASES[phase] : 'The install stopped.' }}</p>

    <div ref="panel" class="output" aria-label="What the install is printing">
      <p v-for="line in lines" :key="line.id" :class="line.stream">{{ line.text }}</p>
    </div>

    <Message v-if="failure" tone="error" :text="failure.text" :detail="failure.detail" />

    <div class="actions">
      <button v-if="running && STOPPABLE.includes(phase)" type="button" @click="stop">
        Cancel
      </button>
      <template v-else-if="!running">
        <button
          v-if="hasAccountAlready"
          type="button"
          class="primary"
          @click="emit('sign-in', host, port)"
        >
          Sign in instead
        </button>
        <button v-else type="button" class="primary" @click="run">Try again</button>
        <button type="button" @click="emit('back')">Back</button>
      </template>
    </div>
  </template>
</template>
