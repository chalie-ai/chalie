import '@fontsource/inter/300.css';
import '@fontsource/inter/400.css';
import '@fontsource/inter/500.css';
import '@fontsource/inter/600.css';
import '@chalie/shared/styles/main.scss';
import './styles/interface.scss';
import { createApp } from 'vue';
import { createPinia } from 'pinia';
import App from './App.vue';
import { router, authGateRedirected } from './router';

const app = createApp(App).use(createPinia()).use(router);

// Gate the mount behind the auth gate. `router.isReady()` resolves once the
// initial navigation — including the async `beforeEach` gate — has settled, so
// by the time it fires we know whether the gate issued a hard redirect. Mounting
// only when it did not keeps App.vue's onMounted (and therefore the WebSocket
// connect in session.init()) from firing on a page that is navigating away.
// Parity with legacy app.js:58 (`await chalieGateReady; if (!gate.stay) return;`).
router.isReady().finally(() => {
  if (!authGateRedirected()) app.mount('#app');
});
