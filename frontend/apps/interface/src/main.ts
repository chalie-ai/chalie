import './styles/fonts.css';
import '@chalie/shared/styles/main.scss';
import './styles/interface.scss';
import './styles/conversation.scss';
import './styles/rich-card-base.scss';
import { createApp } from 'vue';
import { createPinia } from 'pinia';
import App from './App.vue';
import { router, authGateRedirected } from './router';

const app = createApp(App).use(createPinia()).use(router);

// Gate the mount behind the auth gate. `router.isReady()` resolves after the async
// `beforeEach` gate settles, so we know whether it issued a hard redirect. Mounting
// only when it did not keeps App.vue's onMounted (the WebSocket connect in
// session.init()) from firing on a page that is navigating away.
router.isReady().finally(() => {
  if (!authGateRedirected()) app.mount('#app');
});
