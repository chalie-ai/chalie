import '../styles/fonts.css';
import '@chalie/shared/styles/main.scss';
import './login.scss';
import { createApp } from 'vue';
import { createPinia } from 'pinia';
import LoginPage from './LoginPage.vue';
import { system } from '../api/system';

// Pre-mount auth gate: existing session → redirect to / and skip mounting.
(async () => {
  try {
    const status = await system.authStatus();
    if (status.has_session) {
      window.location.replace('/');
      return;
    }
  } catch {
    // Failed status read → stay and mount (treated as "not signed in").
  }
  createApp(LoginPage).use(createPinia()).mount('#app');
})();
