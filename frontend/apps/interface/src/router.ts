import { createRouter, createWebHistory } from 'vue-router';
import HomeView from './views/HomeView.vue';
import { system } from './api/system';

export const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [{ path: '/', name: 'home', component: HomeView }],
});

/**
 * Auth gate — port of frontend/shared/auth-gate.js `chat` page branch.
 *
 * Runs once on the initial navigation to `/`. On network error the gate
 * stays (server guards anyway). On missing account/session/providers it
 * performs a hard redirect to the appropriate legacy route.
 */
router.beforeEach(async () => {
  try {
    const status = await system.authStatus();
    const { has_master_account, has_session, has_providers } = status;

    if (!has_master_account) {
      window.location.replace('/on-boarding/');
      return false;
    }
    if (!has_session) {
      window.location.replace(
        '/login/?next=' + encodeURIComponent(window.location.pathname + window.location.search),
      );
      return false;
    }
    if (!has_providers) {
      window.location.replace('/brain/');
      return false;
    }
    return true;
  } catch {
    // Network / server down — stay; server guards anyway.
    return true;
  }
});
