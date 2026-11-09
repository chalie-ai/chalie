<!-- Slide-over browser panel: the server browser's tabs, one shown live, with
     an address bar and back, forward and reload. Chalie and the user act on a
     tab freely; when Chalie hands one over (a login, a captcha) a bar says why
     and Done gives it back. Escape closes the panel, except while the page
     has the keys; F6 moves the focus between the page and the toolbar, and
     Tab stays inside the panel. -->
<script setup lang="ts">
import { computed, nextTick, onUpdated, ref, watch } from 'vue';
import { ArrowLeft, ArrowRight, Keyboard, Plus, RotateCw, X } from '@lucide/vue';
import { useBrowserStore } from '../../stores/browser';
import type { BrowserTab } from '../../api/browser';
import { withScheme, type StreamMeta } from '../../utils/browserInput';
import BrowserView from './BrowserView.vue';

const browser = useBrowserStore();

const panelRef = ref<HTMLElement | null>(null);
const addressRef = ref<HTMLInputElement | null>(null);
const bodyRef = ref<HTMLElement | null>(null);
const viewRef = ref<InstanceType<typeof BrowserView> | null>(null);
/** The shown tab's back and forward, from its stream; unknown until the stream says. */
const canGo = ref<{ back: boolean; forward: boolean }>({ back: false, forward: false });
const address = ref('');
/** The empty state's address field. */
const firstAddress = ref('');
/** The user asked for the on-screen keyboard; the toggle shows only on touch screens. */
const keyboard = ref(false);
/** What had focus when the panel opened; it gets focus back on close. */
let opener: HTMLElement | null = null;
let editingAddress = false;

const tab = computed(() => browser.selectedTab);

/** A blank tab reports 'about:blank' as its url and title. */
function tabName(t: BrowserTab): string {
  return t.url === 'about:blank' ? 'New tab' : t.title || t.url;
}

/** A blank tab shows an empty address field, as a new tab does in any browser. */
function shownAddress(url: string | undefined): string {
  return url === 'about:blank' ? '' : (url ?? '');
}

watch(
  () => browser.open,
  async (isOpen) => {
    if (isOpen) {
      opener = document.activeElement instanceof HTMLElement ? document.activeElement : null;
      resetAddress();
      await nextTick();
      panelRef.value?.focus();
      return;
    }
    if (opener?.isConnected) opener.focus();
    opener = null;
  },
);

// An address typed for one tab is never sent to the next.
watch(
  () => browser.selectedTabId,
  () => {
    if (document.activeElement === addressRef.value) panelRef.value?.focus();
    resetAddress();
    keyboard.value = false;
    canGo.value = { back: false, forward: false };
  },
);

// A control that had the focus went away (Done, a tab's close, the tab
// itself): the focus stays in the panel rather than falling to the page
// behind, where Escape would close what is open under it.
onUpdated(() => {
  if (browser.open && document.activeElement === document.body) panelRef.value?.focus();
});

// The address bar follows the page, but never over what the user is typing.
watch(
  () => shownAddress(tab.value?.url),
  (url) => {
    if (!editingAddress) address.value = url;
  },
  { immediate: true },
);

function onMeta(meta: StreamMeta): void {
  canGo.value = { back: meta.can_go_back, forward: meta.can_go_forward };
}

function onAddressFocus(): void {
  editingAddress = true;
  addressRef.value?.select();
}

function resetAddress(): void {
  editingAddress = false;
  address.value = shownAddress(tab.value?.url);
}

function go(): void {
  const current = tab.value;
  const typed = address.value.trim();
  if (!current || !typed) return;
  const url = withScheme(typed);
  viewRef.value?.focusPage();
  // Shown as sent until the page reports where it landed.
  address.value = url;
  void browser.navigate(current.id, url);
}

// Synchronous, so the field is focused inside the tap and the phone raises its keyboard.
function toggleKeyboard(): void {
  keyboard.value = !keyboard.value;
  viewRef.value?.setKeyboard(keyboard.value);
  if (!keyboard.value) panelRef.value?.focus();
}

async function newTab(): Promise<void> {
  if (!(await browser.createTab())) return;
  await nextTick();
  addressRef.value?.focus();
}

async function openFirst(): Promise<void> {
  const typed = firstAddress.value.trim();
  if (typed && (await browser.createTab(withScheme(typed)))) firstAddress.value = '';
}

/** F6: from the page to the toolbar, and from anywhere else in the panel to the page. */
function switchFocus(): void {
  if (bodyRef.value?.contains(document.activeElement)) addressRef.value?.focus();
  else viewRef.value?.focusPage();
}

/** Tab past the last control comes back to the first, and Shift+Tab the other way. */
function containTab(e: KeyboardEvent): void {
  const panel = panelRef.value;
  if (!panel) return;
  const controls = [
    ...panel.querySelectorAll<HTMLElement>('button:not(:disabled), input:not(:disabled), textarea'),
  ].filter((el) => el.getClientRects().length > 0);
  const first = controls[0];
  const last = controls.at(-1);
  if (!first || !last) return;
  const active = document.activeElement;
  if (e.shiftKey && (active === first || active === panel)) {
    e.preventDefault();
    last.focus();
  } else if (!e.shiftKey && active === last) {
    e.preventDefault();
    first.focus();
  }
}

function onKeydown(e: KeyboardEvent): void {
  switch (e.key) {
    case 'Escape':
      // Stop here, so a thread panel open underneath stays open.
      e.stopPropagation();
      browser.closePanel();
      return;
    case 'F6':
      e.preventDefault();
      e.stopPropagation();
      switchFocus();
      return;
    case 'Tab':
      containTab(e);
  }
}
</script>

<template>
  <Transition name="browser-panel">
    <aside
      v-if="browser.open"
      ref="panelRef"
      class="browser-panel"
      role="dialog"
      aria-modal="true"
      aria-label="Browser"
      tabindex="-1"
      @keydown="onKeydown"
    >
      <header class="browser-panel__header">
        <div class="browser-panel__tabs">
          <div
            v-for="t in browser.tabs"
            :key="t.id"
            class="browser-panel__tab"
            :class="{ 'browser-panel__tab--selected': t.id === browser.selectedTabId }"
          >
            <button
              type="button"
              class="browser-panel__tab-name"
              :aria-current="t.id === browser.selectedTabId ? 'true' : undefined"
              :aria-label="t.handoff ? `${tabName(t)}, waiting for you` : undefined"
              :title="tabName(t)"
              @click="browser.select(t.id)"
            >
              <span v-if="t.handoff" class="browser-panel__tab-mark" aria-hidden="true" />
              <span class="browser-panel__tab-text">{{ tabName(t) }}</span>
            </button>
            <button
              v-if="t.opener === 'user'"
              type="button"
              class="btn-icon btn-icon--small"
              :aria-label="`Close ${tabName(t)}`"
              title="Close tab"
              @click="browser.closeTab(t.id)"
            >
              <X :size="16" aria-hidden="true" />
            </button>
          </div>
          <button
            type="button"
            class="btn-icon"
            aria-label="New tab"
            title="New tab"
            @click="newTab"
          >
            <Plus :size="16" aria-hidden="true" />
          </button>
        </div>
        <button
          type="button"
          class="btn-icon"
          aria-label="Close browser"
          title="Close browser"
          @click="browser.closePanel()"
        >
          <X :size="16" aria-hidden="true" />
        </button>
      </header>

      <div v-if="tab" class="browser-panel__toolbar">
        <button
          type="button"
          class="btn-icon browser-panel__nav"
          aria-label="Back"
          title="Back"
          :disabled="!canGo.back"
          @click="browser.goHistory(tab.id, 'back')"
        >
          <ArrowLeft :size="16" aria-hidden="true" />
        </button>
        <button
          type="button"
          class="btn-icon browser-panel__nav"
          aria-label="Forward"
          title="Forward"
          :disabled="!canGo.forward"
          @click="browser.goHistory(tab.id, 'forward')"
        >
          <ArrowRight :size="16" aria-hidden="true" />
        </button>
        <button
          type="button"
          class="btn-icon browser-panel__nav"
          aria-label="Reload"
          title="Reload"
          @click="browser.goHistory(tab.id, 'reload')"
        >
          <RotateCw :size="16" aria-hidden="true" />
        </button>
        <form class="browser-panel__address" @submit.prevent="go">
          <input
            ref="addressRef"
            v-model="address"
            type="text"
            inputmode="url"
            enterkeyhint="go"
            autocomplete="off"
            autocapitalize="off"
            spellcheck="false"
            aria-label="Address"
            placeholder="Enter an address"
            @focus="onAddressFocus"
            @blur="resetAddress"
          />
        </form>
        <button
          type="button"
          class="btn-icon browser-panel__keyboard"
          aria-label="Keyboard"
          title="Keyboard"
          :aria-pressed="keyboard"
          @click="toggleKeyboard"
        >
          <Keyboard :size="16" aria-hidden="true" />
        </button>
      </div>

      <div v-if="tab?.handoff" class="browser-panel__handoff" role="status">
        <span class="browser-panel__handoff-text">
          Chalie needs you here{{ tab.handoff.reason ? `: ${tab.handoff.reason}` : '' }}
        </span>
        <button type="button" class="btn btn-primary" @click="browser.finishHandoff(tab.id)">
          Done
        </button>
      </div>

      <div ref="bodyRef" class="browser-panel__body">
        <BrowserView v-if="tab" ref="viewRef" :key="tab.id" :tab-id="tab.id" @meta="onMeta" />
        <form v-else class="browser-panel__empty" @submit.prevent="openFirst">
          <p class="browser-panel__empty-title">No tabs open</p>
          <div class="browser-panel__empty-row">
            <input
              v-model="firstAddress"
              type="text"
              inputmode="url"
              enterkeyhint="go"
              autocomplete="off"
              autocapitalize="off"
              spellcheck="false"
              aria-label="Address"
              placeholder="Enter an address"
            />
            <button type="submit" class="btn btn-primary" :disabled="!firstAddress.trim()">
              Open
            </button>
          </div>
        </form>
      </div>
    </aside>
  </Transition>
</template>

<style scoped lang="scss">
// At most 1280px and 95% of the window wide; the whole window on a phone. The
// page's window is whatever size this leaves the view. The panel takes the
// focus only to hold the keys while no control has it, so it draws no ring.
.browser-panel {
  position: fixed;
  top: 56px;
  right: 0;
  bottom: 0;
  width: min(95%, 1280px);
  z-index: 130;
  display: flex;
  flex-direction: column;
  background: var(--surface);
  overflow: hidden;
  outline: none;
}

@media (max-width: 640px) {
  .browser-panel {
    width: 100%;
  }
}

.browser-panel__header {
  flex-shrink: 0;
  display: flex;
  align-items: center;
  gap: var(--space-sm);
  height: 46px;
  padding: 0 var(--space-sm);
}

.browser-panel__tabs {
  flex: 1;
  min-width: 0;
  display: flex;
  align-items: center;
  gap: var(--space-xs);
  overflow-x: auto;
}

.browser-panel__tab {
  flex-shrink: 0;
  display: flex;
  align-items: center;
  max-width: 220px;
  height: var(--control-h);
  color: var(--muted);
  transition:
    background-color var(--dur-1) var(--ease-out),
    color var(--dur-1) var(--ease-out);
}

.browser-panel__tab:hover {
  color: var(--text);
}

.browser-panel__tab--selected {
  background: var(--surface-2);
  color: var(--text);
}

.browser-panel__tab-name {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
  min-width: 0;
  height: 100%;
  padding: 0 var(--space-sm);
  background: none;
  color: inherit;
  font: 500 var(--fs-body) var(--font-ui);
}

.browser-panel__tab-text {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

// The tab Chalie handed over.
.browser-panel__tab-mark {
  flex-shrink: 0;
  width: 8px;
  height: 8px;
  background: var(--pink);
}

.browser-panel__toolbar {
  flex-shrink: 0;
  display: flex;
  align-items: center;
  gap: var(--space-xs);
  padding: 0 var(--space-sm) var(--space-sm);
}

.browser-panel__nav:disabled {
  color: var(--muted);
  opacity: 0.45;
}

.browser-panel__address {
  flex: 1;
  min-width: 0;
}

.browser-panel__address input {
  width: 100%;
}

.browser-panel__keyboard[aria-pressed='true'] {
  background: var(--surface-2);
  color: var(--text);
}

@media not all and (pointer: coarse) {
  .browser-panel__keyboard {
    display: none;
  }
}

.browser-panel__handoff {
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-md);
  margin: 0 var(--space-sm) var(--space-sm);
  padding: var(--space-sm) var(--space-sm) var(--space-sm) var(--space-md);
  background: var(--surface-2);
  color: var(--text);
}

.browser-panel__handoff-text {
  min-width: 0;
}

.browser-panel__body {
  position: relative;
  flex: 1;
  min-height: 0;
  background: var(--bg);
}

.browser-panel__empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: var(--space-md);
  height: 100%;
  padding: var(--space-lg);
}

.browser-panel__empty-title {
  margin: 0;
  font: 600 var(--fs-title) var(--font-ui);
  color: var(--text);
}

.browser-panel__empty-row {
  display: flex;
  gap: var(--space-sm);
  width: min(480px, 100%);
}

.browser-panel__empty-row input {
  flex: 1;
  min-width: 0;
}

// The panel slides in on open and plays the same slide backwards on close.
.browser-panel-enter-active,
.browser-panel-leave-active {
  animation: slide-in var(--dur-3) var(--ease-out);
}

.browser-panel-leave-active {
  animation-direction: reverse;
  animation-fill-mode: forwards;
}
</style>
