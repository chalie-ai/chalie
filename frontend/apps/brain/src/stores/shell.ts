import { defineStore } from 'pinia';

/**
 * Shell-level state for the Brain app.
 *
 * Port of the module-scoped state in legacy app.js:
 *   _providersOnly, _sidebarCollapsed, _mobileOpen, _heartbeatFired.
 *
 * Options-style Pinia store (consistent with legacy OO module style).
 */
export const useShellStore = defineStore('brain-shell', {
  state: () => ({
    /** When true: the app launched with no providers configured. Navigation is
     *  locked to the Providers panel until a provider is added. No hard redirect —
     *  the Brain app stays mounted (legacy auth-gate.js:65-67 brain branch). */
    providersOnly: false as boolean,

    /** Whether the desktop sidebar is in collapsed (icon-only) mode. */
    sidebarCollapsed: false as boolean,

    /** Whether the mobile sidebar overlay is visible. */
    mobileOpen: false as boolean,

    /** Whether the command palette overlay is open. */
    commandPaletteOpen: false as boolean,
  }),

  actions: {
    /**
     * Called by the Providers panel after a provider is successfully saved and
     * confirmed via GET /auth/status → has_providers=true.
     *
     * Lifts the navigation lock so the sidebar enables all items.
     * Port of legacy BrainApp.liftProvidersOnly().
     */
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
