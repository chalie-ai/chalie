import '@fontsource/inter/300.css';
import '@fontsource/inter/400.css';
import '@fontsource/inter/500.css';
import '@fontsource/inter/600.css';
import '@chalie/shared/styles/main.scss';
import './styles/brain.scss';
import { createApp } from 'vue';
import { createPinia } from 'pinia';
import App from './App.vue';
import { router, authGateRedirected } from './router';

const app = createApp(App).use(createPinia()).use(router);

// Gate the mount behind the auth gate. router.isReady() resolves once the
// initial navigation — including the async beforeEach guard — has settled.
// If the gate issued a hard redirect, do not mount so the SPA shell does not
// flash before the page navigates away.
// Parity with legacy app.js:351 (`await chalieGateReady; if (!gate.stay) return;`).
router.isReady().finally(() => {
  if (!authGateRedirected()) app.mount('#app');
});
