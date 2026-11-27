import '@chalie/shared/styles/main.scss';
import './styles/interface.scss';
import './styles/conversation.scss';
import './styles/rich-card-base.scss';
import { createApp } from 'vue';
import { createPinia } from 'pinia';
import { AuthError, HttpError, redirectToLogin, registerReplyFailureReporter } from '@chalie/shared';
import App from './App.vue';
import { system } from './api/system';
import type { AuthStatus } from './api/system';
import { registerAppContext } from './utils/turnDom';
import { showToast } from './utils/toast';

const app = createApp(App).use(createPinia());
registerAppContext(app);
registerReplyFailureReporter((message) => showToast(message));

/**
 * The sign-in gate: says whether the chat may load, and sends anyone who may
 * not use it on. A server that answers the check with an error counts as no
 * account; one that cannot be reached lets the chat load.
 */
async function signInGate(): Promise<boolean> {
  let status: AuthStatus;
  try {
    status = await system.authStatus();
  } catch (err) {
    if (!(err instanceof HttpError || err instanceof AuthError)) return true;
    window.location.replace('/on-boarding/');
    return false;
  }
  if (!status.has_master_account) window.location.replace('/on-boarding/');
  else if (!status.has_session) redirectToLogin();
  else if (!status.has_providers) window.location.replace('/brain/');
  else return true;
  return false;
}

// Mount only once the gate lets the chat load, so App.vue's onMounted (the
// socket connect in session.init()) never runs on a page that is navigating away.
void signInGate().then((mayLoad) => {
  if (mayLoad) app.mount('#app');
});
