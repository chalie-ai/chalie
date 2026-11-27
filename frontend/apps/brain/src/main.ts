import '@chalie/shared/styles/main.scss';
import './styles/brain.scss';
import { createApp } from 'vue';
import { createPinia } from 'pinia';
import { registerReplyFailureReporter } from '@chalie/shared';
import App from './App.vue';
import { authGateRedirected, router } from './router';
import { useToast } from './composables/useToast';

const app = createApp(App).use(createPinia()).use(router);
registerReplyFailureReporter((message) => useToast().show(message, 'error'));

// isReady() resolves after the async beforeEach gate settles. Skip mount on a
// hard redirect so the shell never flashes before navigating away.
router.isReady().finally(() => {
  if (!authGateRedirected()) app.mount('#app');
});
