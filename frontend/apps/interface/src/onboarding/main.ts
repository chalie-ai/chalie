import '@fontsource/inter/300.css';
import '@fontsource/inter/400.css';
import '@fontsource/inter/500.css';
import '@fontsource/inter/600.css';
import '@chalie/shared/styles/main.scss';
import './onboarding.scss';
import { createApp } from 'vue';
import { createPinia } from 'pinia';
import OnboardingPage from './OnboardingPage.vue';
import { system } from '../api/system';

// Pre-mount auth gate: existing master account → redirect to /login/ and skip
// mounting. On failure, stay and mount so an outage doesn't lock the user out of setup.
(async () => {
  try {
    const status = await system.authStatus();
    if (status.has_master_account) {
      window.location.replace('/login/');
      return;
    }
  } catch {
    // Failed status read → stay and mount.
  }
  createApp(OnboardingPage).use(createPinia()).mount('#app');
})();
