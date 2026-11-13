import { defineStore } from 'pinia';

/** Shell-level UI state for the Brain app. */
export const useShellStore = defineStore('brain-shell', {
  state: () => ({
    // No providers configured: nav locked to Providers panel, app stays mounted (no hard redirect).
    providersOnly: false as boolean,
    sidebarCollapsed: false as boolean,
    mobileOpen: false as boolean,
    commandPaletteOpen: false as boolean,
  }),

  actions: {
    // Called once a provider is saved + confirmed via /auth/status; lifts the nav lock.
    liftProvidersOnly(): void {
      this.providersOnly = false;
    },

    openCommandPalette(): void {
      this.commandPaletteOpen = true;
    },

    closeCommandPalette(): void {
      this.commandPaletteOpen = false;
    },

    toggleCommandPalette(): void {
      this.commandPaletteOpen = !this.commandPaletteOpen;
    },

    toggleSidebar(): void {
      this.sidebarCollapsed = !this.sidebarCollapsed;
    },

    openMobileSidebar(): void {
      this.mobileOpen = true;
    },

    closeMobileSidebar(): void {
      this.mobileOpen = false;
    },
  },
});
