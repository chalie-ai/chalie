import '@fontsource/inter/300.css';
import '@fontsource/inter/400.css';
import '@fontsource/inter/500.css';
import '@fontsource/inter/600.css';
import '@chalie/shared/styles/main.scss';
import './onboarding.scss';
import { createApp } from 'vue';
import { createPinia } from 'pinia';
import OnboardingPage from './OnboardingPage.vue';
import { auth } from '../api/auth';

function mount(): void {
  createApp(OnboardingPage).use(createPinia()).mount('#app');
}

// Pre-mount auth gate: if a master account already exists, redirect to /login/
// and skip mounting entirely. Parity with auth-gate.js page='onboarding' rule:
//   "onboarding : account → /login/"
// On network failure, stay on the page and mount so an outage doesn't lock
// the user out of setup.
(async () => {
  try {
    const status = await auth.authStatus();
    if (status.has_master_account) {
      window.location.replace('/login/');
      // Do not mount — the page is navigating away.
      return;
    }
  } catch {
    // Any failure to read status (network error or non-2xx) → stay and mount,
    // matching auth-gate.js, which treats a failed status read as "stay".
  }
  mount();
})();
