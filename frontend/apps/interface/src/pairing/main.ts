import '../styles/fonts.css';
import '@chalie/shared/styles/main.scss';
import { createApp } from 'vue';
import { getToken } from '@chalie/shared';
import LinkDevice from './LinkDevice.vue';

// Native-only QR pairing page (separate MPA entry, like /login/). The router gate
// redirects the unpaired Tauri app here via a full document load, so the chat
// SPA's gate never runs on this page. Already paired (token present) → there is
// nothing to do, so bounce to the app.
if (getToken()) {
  globalThis.location.replace('/');
} else {
  createApp(LinkDevice).mount('#app');
}
