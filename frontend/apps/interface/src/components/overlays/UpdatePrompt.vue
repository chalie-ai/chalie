<script setup lang="ts">
/**
 * UpdatePrompt — update-available banner + detail dialog.
 *
 * Port of frontend/interface/update_system.js.
 *
 * DORMANT: the backend does not currently emit `app_update` WS events. This
 * component is ported for parity and will activate automatically when the
 * backend ships the event.
 *
 * State is owned by useNotificationsStore:
 *   Banner dismiss  → notifications.dismissUpdate()
 *   Apply           → notifications.applyUpdate()
 *   Dialog open     → showDialog = true (local, no store needed)
 *   Dialog close    → showDialog = false
 *
 * Deployment-mode instructions mirror update_system.js _applyDialogDeploymentMode():
 *   'docker' — docker pull + compose up instructions
 *   'dev'    — git pull origin main instruction
 *   other    — Apply button offered
 */
import { ref, computed, watch, onBeforeUnmount } from 'vue';
import { useNotificationsStore } from '../../stores/notifications';

const notifications = useNotificationsStore();

const update = computed(() => notifications.currentUpdate);
const visible = computed(() => update.value !== null);

// ── Local dialog state ────────────────────────────────────────────────────────

const showDialog = ref(false);

function openDialog(): void {
  if (!update.value) return;
  showDialog.value = true;
}

function closeDialog(): void {
  showDialog.value = false;
}

// ── Computed helpers ──────────────────────────────────────────────────────────

/** True when the deployment mode does not support one-click apply. */
const isManualMode = computed<boolean>(() => {
  const mode = update.value?.deployment_mode;
  return mode === 'docker' || mode === 'dev';
});

/** Banner action button label — port of update_system.js _showUpdateBanner() lines 34-37. */
const bannerActionLabel = computed<string>(() => isManualMode.value ? 'Details' : 'Update');

// ── Apply ─────────────────────────────────────────────────────────────────────

async function applyUpdate(): Promise<void> {
  await notifications.applyUpdate();
}

// ── Dismiss ───────────────────────────────────────────────────────────────────

function dismissBanner(): void {
  notifications.dismissUpdate();
  closeDialog();
}

// ── Escape-to-close for <dialog open> (D9) ────────────────────────────────────
// Legacy update_system.js used native <dialog>.showModal() which handles Escape
// automatically. The Vue port uses <dialog open> (a static attribute), so the
// browser @cancel event never fires. Wire a manual keydown handler instead.

function onKeydown(e: KeyboardEvent): void {
  if (e.key === 'Escape' && showDialog.value) {
    closeDialog();
  }
}

// Add listener only while the component is alive; tear down on unmount.
document.addEventListener('keydown', onKeydown);

onBeforeUnmount(() => {
  document.removeEventListener('keydown', onKeydown);
});

// Keep the watcher in sync: if the banner is dismissed externally while the
// dialog is open, close the dialog too.
watch(visible, (v) => {
  if (!v) closeDialog();
});
</script>

<template>
  <template v-if="visible && update">
    <!--
      Update banner — port of #updateBanner from the legacy HTML + style.css lines 208-271.
      Slides in from the top below the presence bar (top: 56px).
    -->
    <Transition name="update-banner">
      <div v-if="visible" class="update-banner">
        <!-- Icon -->
        <span class="update-banner__icon" aria-hidden="true">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="16 16 12 12 8 16" />
            <line x1="12" y1="12" x2="12" y2="21" />
            <path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3" />
          </svg>
        </span>

        <!-- Version text — port of #updateBannerVersion text content -->
        <span class="update-banner__text">
          New version available: <strong>v{{ update.latest_version }}</strong>
        </span>

        <!-- Action button: "Details" for manual modes, "Update" otherwise -->
        <button class="update-banner__action" @click="openDialog">
          {{ bannerActionLabel }}
        </button>

        <!-- Dismiss ✕ -->
        <button
          class="update-banner__dismiss"
          aria-label="Dismiss update notification"
          @click="dismissBanner"
        >
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" aria-hidden="true">
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </button>
      </div>
    </Transition>

    <!--
      Update dialog — port of #updateDialog <dialog> element.
      Uses a native <dialog> element to match the legacy showModal() / close() API.
    -->
    <Transition name="update-dialog">
      <dialog
        v-if="showDialog"
        class="update-dialog"
        open
        aria-modal="true"
        aria-label="Update available"
        @cancel.prevent="closeDialog"
      >
        <div class="update-dialog__content">
          <!-- Close button — port of #updateDialogClose -->
          <button
            class="update-dialog__close btn-icon"
            aria-label="Close"
            @click="closeDialog"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>

          <!-- Title -->
          <h2 class="update-dialog__title">Update available</h2>

          <!-- Version line — port of #updateCurrentVer / #updateNewVer -->
          <p class="update-dialog__version">
            <strong>v{{ update.current_version }}</strong>
            &nbsp;→&nbsp;
            <strong>v{{ update.latest_version }}</strong>
          </p>

          <!-- Release notes — port of #updateNotes -->
          <div class="update-dialog__notes">
            {{ update.release_notes || 'No release notes.' }}
          </div>

          <!--
            Actions section — port of #updateActions.
            Hidden when deployment mode requires manual steps (docker / dev).
            Also hidden while applying (replaced by progress).
          -->
          <div
            v-if="!isManualMode && !notifications.applyingUpdate && !notifications.updateApplyMessage"
            class="update-dialog__actions"
          >
            <button class="btn btn-secondary" @click="closeDialog">Cancel</button>
            <button class="btn btn-primary" @click="applyUpdate">Apply update</button>
          </div>

          <!--
            Progress section — port of #updateProgress.
            Shown while applying. On failure, store restores actions after 3 s.
          -->
          <div v-if="notifications.applyingUpdate || notifications.updateApplyMessage" class="update-dialog__progress">
            <div v-if="notifications.applyingUpdate" class="update-dialog__spinner" aria-hidden="true" />
            <p class="update-dialog__status">
              {{ notifications.updateApplyMessage ?? 'Applying update...' }}
            </p>
          </div>

          <!--
            Deployment-mode instructions — port of #updateInstructions.
            Shown instead of actions when mode is 'docker' or 'dev'.
          -->
          <div v-if="isManualMode" class="update-dialog__instructions">
            <!-- docker mode — port of _applyDialogDeploymentMode lines 86-94 -->
            <template v-if="update.deployment_mode === 'docker'">
              <p>You're running Chalie in Docker. To update:</p>
              <code>docker pull chalie/chalie:{{ update.latest_tag }}
docker compose up -d</code>
            </template>

            <!-- dev mode — port of _applyDialogDeploymentMode lines 95-98 -->
            <template v-else-if="update.deployment_mode === 'dev'">
              <p>You're running from a git clone. To update:</p>
              <code>git pull origin main</code>
            </template>
          </div>
        </div>
      </dialog>
    </Transition>

    <!-- Dialog backdrop (scrim) when dialog is open -->
    <Transition name="update-scrim">
      <div v-if="showDialog" class="update-dialog-scrim" aria-hidden="true" @click="closeDialog" />
    </Transition>
  </template>
</template>

<style scoped lang="scss">
/*
 * Styles are a faithful port of .update-banner / .update-dialog rules from
 * frontend/interface/style.css (lines 208-394).
 * No hardcoded rgba colors — CSS custom properties only.
 */

// ── Banner (port of .update-banner, lines 208-271) ────────────────────────────

.update-banner {
  position: fixed;
  top: 56px;
  left: 50%;
  transform: translateX(-50%);
  z-index: 1000;
  display: flex;
  align-items: center;
  gap: var(--space-sm);
  padding: var(--space-sm) var(--space-md);
  background: color-mix(in oklab, var(--accent-tertiary) 8%, transparent);
  border: 1px solid color-mix(in oklab, var(--accent-tertiary) 25%, transparent);
  border-radius: var(--radius-full);
  font-size: var(--font-size-sm);
  color: var(--accent-tertiary);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
}

.update-banner__icon {
  display: flex;
  align-items: center;
  color: var(--accent-tertiary);
}

.update-banner__text {
  color: var(--text-primary);

  strong {
    color: var(--accent-tertiary);
  }
}

.update-banner__action {
  background: var(--accent-tertiary);
  color: var(--bg);
  border: none;
  border-radius: var(--radius-full);
  padding: 2px 12px;
  font-size: var(--font-size-sm);
  font-weight: 500;
  cursor: pointer;
  transition: opacity var(--duration-fast) var(--ease-out);

  &:hover {
    opacity: 0.85;
  }
}

.update-banner__dismiss {
  background: none;
  border: none;
  color: var(--text-tertiary);
  cursor: pointer;
  padding: 2px;
  display: flex;
  align-items: center;
  transition: color var(--duration-fast) var(--ease-out);

  &:hover {
    color: var(--text-primary);
  }
}

// ── Dialog (port of .update-dialog / .update-dialog__content, lines 273-295) ─

.update-dialog {
  background: transparent;
  border: none;
  padding: 0;
  max-width: 480px;
  width: 90vw;
  // Positioned manually — we use a scrim div instead of ::backdrop for
  // Vue Transition support (native ::backdrop cannot be transitioned via JS).
  position: fixed;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  z-index: 1200;
}

.update-dialog__content {
  position: relative;
  background: var(--surface-elevated, rgba(16, 20, 32, 0.98));
  border: 1px solid color-mix(in oklab, var(--accent-tertiary) 15%, transparent);
  border-radius: var(--radius-md);
  padding: var(--space-lg);
  color: var(--text-primary);
}

.update-dialog__close {
  position: absolute;
  top: var(--space-md);
  right: var(--space-md);
  color: var(--text-secondary);

  &:hover {
    color: var(--text-primary);
  }
}

// ── Dialog body elements ───────────────────────────────────────────────────────

.update-dialog__title {
  font-size: var(--font-size-lg);
  font-weight: 500;
  margin: 0 0 var(--space-sm);
  color: var(--accent-tertiary);
}

.update-dialog__version {
  font-size: var(--font-size-sm);
  color: var(--text-secondary);
  margin: 0 0 var(--space-md);

  strong {
    color: var(--accent-tertiary);
  }
}

.update-dialog__notes {
  max-height: 200px;
  overflow-y: auto;
  padding: var(--space-md);
  background: color-mix(in oklab, var(--text) 3%, transparent);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-sm);
  font-size: var(--font-size-sm);
  color: var(--text-secondary);
  line-height: 1.6;
  margin-bottom: var(--space-md);
  white-space: pre-wrap;
}

.update-dialog__actions {
  display: flex;
  justify-content: flex-end;
  gap: var(--space-sm);
}

// ── Progress (port of .update-dialog__progress / __spinner / __status) ────────

.update-dialog__progress {
  display: flex;
  align-items: center;
  gap: var(--space-md);
  padding: var(--space-md) 0;
}

.update-dialog__spinner {
  width: 20px;
  height: 20px;
  border: 2px solid color-mix(in oklab, var(--accent-tertiary) 20%, transparent);
  border-top-color: var(--accent-tertiary);
  border-radius: 50%;
  flex-shrink: 0;
  animation: update-spin 0.8s linear infinite;
}

@keyframes update-spin {
  to { transform: rotate(360deg); }
}

.update-dialog__status {
  font-size: var(--font-size-sm);
  color: var(--accent-tertiary);
  margin: 0;
}

// ── Instructions (port of .update-dialog__instructions / code) ───────────────

.update-dialog__instructions {
  padding: var(--space-md);
  background: color-mix(in oklab, var(--text) 3%, transparent);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-sm);
  font-size: var(--font-size-sm);
  color: var(--text-secondary);
  line-height: 1.6;

  p {
    margin: 0 0 var(--space-sm);
  }

  code {
    display: block;
    background: color-mix(in oklab, var(--text) 6%, transparent);
    padding: 8px 10px;
    border-radius: 4px;
    font-size: 0.8rem;
    white-space: pre;
  }
}

// ── Backdrop scrim ────────────────────────────────────────────────────────────

.update-dialog-scrim {
  position: fixed;
  inset: 0;
  z-index: 1100;
  background: var(--overlay-scrim, rgba(6, 8, 14, 0.75));
  backdrop-filter: blur(4px);
}

// ── Vue Transitions ───────────────────────────────────────────────────────────

.update-banner-enter-active,
.update-banner-leave-active {
  transition:
    opacity var(--duration-normal, 200ms) var(--ease-out, ease),
    transform var(--duration-normal, 200ms) var(--ease-out, ease);
}

.update-banner-enter-from,
.update-banner-leave-to {
  opacity: 0;
  transform: translateX(-50%) translateY(-8px);
}

.update-dialog-enter-active,
.update-dialog-leave-active {
  transition:
    opacity var(--duration-normal, 200ms) var(--ease-out, ease),
    transform var(--duration-normal, 200ms) var(--ease-out, ease);
}

.update-dialog-enter-from,
.update-dialog-leave-to {
  opacity: 0;
  transform: translate(-50%, calc(-50% - 8px));
}

.update-scrim-enter-active,
.update-scrim-leave-active {
  transition: opacity var(--duration-normal, 200ms) var(--ease-out, ease);
}

.update-scrim-enter-from,
.update-scrim-leave-to {
  opacity: 0;
}
</style>
