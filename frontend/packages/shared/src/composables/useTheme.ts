import { storeToRefs } from 'pinia';
import { useThemeStore } from '../stores/theme';

export function useTheme() {
  const store = useThemeStore();
  const { theme } = storeToRefs(store);
  return { theme, setTheme: store.setTheme, toggle: store.toggle, init: store.init };
}
