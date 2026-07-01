<script setup lang="ts">
/**
 * UpdatePrompt — update-available banner + detail dialog.
 *
 * DORMANT: the backend does not currently emit `app_update` WS events; ported
 * for parity, activates automatically when the backend ships the event.
 *
 * State is owned by useNotificationsStore (dialog open/close is local). Manual
 * deployment modes ('docker' / 'dev') show instructions instead of an Apply button.
 */
import { computed, onBeforeUnmount, ref, watch } from 'vue';
import { CloudDownload, X } from '@lucide/vue';
import { useNotificationsStore } from '../../stores/notifications';

const notifications = useNotificationsStore();

const update = computed(() => notifications.currentUpdate);
const visible = computed(() => update.value !== null);

const showDialog = ref(false);

function openDialog(): void {
  if (!update.value) return;
  showDialog.value = true;
}

function closeDialog(): void {
  showDialog.value = false;
}

/** True when the deployment mode does not support one-click apply. */
const isManualMode = computed<boolean>(() => {
  const mode = update.value?.deployment_mode;
  return mode === 'docker' || mode === 'dev';
});

function dismissBanner(): void {
  notifications.dismissUpdate();
  closeDialog();
}

// <dialog open> (static attribute) never fires the browser @cancel event, so
// Escape must be wired manually here.
function onKeydown(e: KeyboardEvent): void {
  if (e.key === 'Escape' && showDialog.value) {
    closeDialog();
  }
}

document.addEventListener('keydown', onKeydown);

onBeforeUnmount(() => {
  document.removeEventListener('keydown', onKeydown);
});

// If the banner is dismissed externally while the dialog is open, close it too.
watch(visible, (v) => {
  if (!v) closeDialog();
});
</script>

<template>
  <template v-if="visible && update">
    <Transition name="update-banner">
      <div v-if="visible" class="update-banner">
        <span class="update-banner__icon" aria-hidden="true">
          <CloudDownload :size="14" />
        </span>

        <span class="update-banner__text">
          New version available: <strong>v{{ update.latest_version }}</strong>
        </span>

        <!-- "Details" for manual modes, "Update" otherwise -->
        <button class="update-banner__action" @click="openDialog">
          {{ isManualMode ? 'Details' : 'Update' }}
        </button>

        <button
          class="update-banner__dismiss"
          aria-label="Dismiss update notification"
          @click="dismissBanner"
        >
          <X :size="12" :stroke-width="2.4" aria-hidden="true" />
        </button>
      </div>
    </Transition>

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
          <button class="update-dialog__close btn-icon" aria-label="Close" @click="closeDialog">
            <X :size="16" aria-hidden="true" />
          </button>

          <h2 class="update-dialog__title">Update available</h2>

          <p class="update-dialog__version">
            <strong>v{{ update.current_version }}</strong>
            &nbsp;→&nbsp;
            <strong>v{{ update.latest_version }}</strong>
          </p>

          <div class="update-dialog__notes">
            {{ update.release_notes || 'No release notes.' }}
          </div>

          <!-- Hidden in manual modes and while applying (replaced by progress). -->
          <div
            v-if="
              !isManualMode && !notifications.applyingUpdate && !notifications.updateApplyMessage
            "
            class="update-dialog__actions"
          >
            <button class="btn btn-secondary" @click="closeDialog">Cancel</button>
            <button class="btn btn-primary" @click="notifications.applyUpdate()">
              Apply update
            </button>
          </div>

          <!-- Shown while applying; on failure the store restores actions after 3s. -->
          <div
            v-if="notifications.applyingUpdate || notifications.updateApplyMessage"
            class="update-dialog__progress"
          >
            <div
              v-if="notifications.applyingUpdate"
              class="update-dialog__spinner"
              aria-hidden="true"
            />
            <p class="update-dialog__status">
              {{ notifications.updateApplyMessage ?? 'Applying update...' }}
            </p>
          </div>

          <!-- Shown instead of actions when mode is 'docker' or 'dev'. -->
          <div v-if="isManualMode" class="update-dialog__instructions">
            <template v-if="update.deployment_mode === 'docker'">
              <p>You're running Chalie in Docker. To update:</p>
              <code>docker pull chalie/chalie:{{ update.latest_tag }} docker compose up -d</code>
            </template>

            <template v-else-if="update.deployment_mode === 'dev'">
              <p>You're running from a git clone. To update:</p>
              <code>git pull origin main</code>
            </template>
          </div>
        </div>
      </dialog>
    </Transition>

    <Transition name="update-scrim">
      <div v-if="showDialog" class="update-dialog-scrim" aria-hidden="true" @click="closeDialog" />
    </Transition>
  </template>
</template>

<style scoped lang="scss">
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

.update-dialog {
  background: transparent;
  border: none;
  padding: 0;
  max-width: 480px;
  width: 90vw;
  // Positioned manually with a scrim div instead of ::backdrop, which cannot be
  // transitioned via Vue Transition.
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
  animation: spin 0.8s linear infinite;
}

.update-dialog__status {
  font-size: var(--font-size-sm);
  color: var(--accent-tertiary);
  margin: 0;
}

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

.update-dialog-scrim {
  position: fixed;
  inset: 0;
  z-index: 1100;
  background: var(--overlay-scrim, rgba(6, 8, 14, 0.75));
  backdrop-filter: blur(4px);
}

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
