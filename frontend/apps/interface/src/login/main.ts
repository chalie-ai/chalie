import '@fontsource/inter/300.css';
import '@fontsource/inter/400.css';
import '@fontsource/inter/500.css';
import '@fontsource/inter/600.css';
import '@chalie/shared/styles/main.scss';
import './login.scss';
import { createApp } from 'vue';
import { createPinia } from 'pinia';
import LoginPage from './LoginPage.vue';
import { auth } from '../api/auth';

function mount(): void {
  createApp(LoginPage).use(createPinia()).mount('#app');
}

// Pre-mount auth gate: if there is already a session, redirect to / and skip
// mounting entirely. Parity with auth-gate.js page='login' rule:
//   "login : session → /"
// On network failure, stay on the page and mount so the user can still try.
(async () => {
  try {
    const status = await auth.authStatus();
    if (status.has_session) {
      window.location.replace('/');
      // Do not mount — the page is navigating away.
      return;
    }
  } catch {
    // Any failure to read status (network error or non-2xx) → stay and mount,
    // matching auth-gate.js, which treats a failed status read as "not signed in".
  }
  mount();
})();
