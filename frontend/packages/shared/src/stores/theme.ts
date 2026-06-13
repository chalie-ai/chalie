import { defineStore } from 'pinia';

export type Theme = 'light' | 'dark';
const THEME_KEY = 'chalie-theme';

function systemTheme(): Theme {
  return globalThis.matchMedia?.('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
}

export const useThemeStore = defineStore('theme', {
  state: () => ({ theme: ((document.documentElement.dataset.theme as Theme) || 'dark') as Theme }),
  actions: {
    setTheme(t: Theme) {
      this.theme = t;
      document.documentElement.dataset.theme = t;
      try {
        localStorage.setItem(THEME_KEY, t);
      } catch {
        /* ignore */
      }
    },
    toggle() {
      this.setTheme(this.theme === 'dark' ? 'light' : 'dark');
    },
    /** Adopt the pre-paint choice (saved → system). Idempotent. */
    init() {
      let saved: string | null = null;
      try {
        saved = localStorage.getItem(THEME_KEY);
      } catch {
        /* ignore */
      }
      this.setTheme(saved === 'light' || saved === 'dark' ? (saved as Theme) : systemTheme());
    },
  },
});
