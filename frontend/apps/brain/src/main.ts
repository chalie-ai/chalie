import '@fontsource/inter/300.css';
import '@fontsource/inter/400.css';
import '@fontsource/inter/500.css';
import '@fontsource/inter/600.css';
import '@chalie/shared/styles/main.scss';
import './styles/brain.scss';
import { createApp } from 'vue';
import { createPinia } from 'pinia';
import App from './App.vue';
import { authGateRedirected, router } from './router';

const app = createApp(App).use(createPinia()).use(router);

// isReady() resolves after the async beforeEach gate settles. Skip mount on a
// hard redirect so the shell never flashes before navigating away.
router.isReady().finally(() => {
  if (!authGateRedirected()) app.mount('#app');
});
