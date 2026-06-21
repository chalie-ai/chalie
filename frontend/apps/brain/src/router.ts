import { createRouter, createWebHistory } from 'vue-router';
import { AuthError, HttpError } from '@chalie/shared';
import { system } from './api/system';

import ProvidersView from './views/ProvidersView.vue';
import CognitionView from './views/CognitionView.vue';
import SchedulerView from './views/SchedulerView.vue';
import ListsView from './views/ListsView.vue';
import DocumentsView from './views/DocumentsView.vue';
import CapabilitiesView from './views/CapabilitiesView.vue';
import PoliciesView from './views/PoliciesView.vue';
import SkillsView from './views/SkillsView.vue';
import McpView from './views/McpView.vue';
import ImportExportView from './views/ImportExportView.vue';

import MemorySub from './views/cognition/MemorySub.vue';
import AutoResearchSub from './views/cognition/AutoResearchSub.vue';
import ToolsSub from './views/cognition/ToolsSub.vue';
import WorldSub from './views/cognition/WorldSub.vue';
import PersonalitySub from './views/cognition/PersonalitySub.vue';
import ErrorsSub from './views/cognition/ErrorsSub.vue';
import UsageSub from './views/cognition/UsageSub.vue';
import CompactionSub from './views/cognition/CompactionSub.vue';

// Scheduler filter tabs are routed for deep-link / breadcrumb support.
import SchedulerAllSub from './views/scheduler/AllSub.vue';
import SchedulerPendingSub from './views/scheduler/PendingSub.vue';
import SchedulerFiredSub from './views/scheduler/FiredSub.vue';
import SchedulerFailedSub from './views/scheduler/FailedSub.vue';
import SchedulerCancelledSub from './views/scheduler/CancelledSub.vue';

import DocumentsActiveSub from './views/documents/ActiveSub.vue';
import DocumentsProcessingSub from './views/documents/ProcessingSub.vue';
import DocumentsUploadsSub from './views/documents/UploadsSub.vue';
import DocumentsDeletedSub from './views/documents/DeletedSub.vue';

import PoliciesChatSub from './views/policies/ChatSub.vue';
import PoliciesBackgroundSub from './views/policies/BackgroundSub.vue';
import PoliciesExternalSub from './views/policies/ExternalSub.vue';

export const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    { path: '/', redirect: '/providers' },

    { path: '/providers', name: 'providers', component: ProvidersView },

    {
      path: '/cognition',
      name: 'cognition',
      component: CognitionView,
      redirect: '/cognition/memory',
      children: [
        { path: 'memory', name: 'cognition-memory', component: MemorySub },
        { path: 'auto-research', name: 'cognition-auto-research', component: AutoResearchSub },
        { path: 'tools', name: 'cognition-tools', component: ToolsSub },
        { path: 'world', name: 'cognition-world', component: WorldSub },
        { path: 'personality', name: 'cognition-personality', component: PersonalitySub },
        { path: 'errors', name: 'cognition-errors', component: ErrorsSub },
        { path: 'usage', name: 'cognition-usage', component: UsageSub },
        // Legacy omitted 'compaction', coercing it to memory; registered so it resolves.
        { path: 'compaction', name: 'cognition-compaction', component: CompactionSub },
      ],
    },

    {
      path: '/scheduler',
      name: 'scheduler',
      component: SchedulerView,
      redirect: '/scheduler/all',
      children: [
        { path: 'all', name: 'scheduler-all', component: SchedulerAllSub },
        { path: 'pending', name: 'scheduler-pending', component: SchedulerPendingSub },
        { path: 'fired', name: 'scheduler-fired', component: SchedulerFiredSub },
        { path: 'failed', name: 'scheduler-failed', component: SchedulerFailedSub },
        { path: 'cancelled', name: 'scheduler-cancelled', component: SchedulerCancelledSub },
      ],
    },

    { path: '/lists', name: 'lists', component: ListsView },

    {
      path: '/documents',
      name: 'documents',
      component: DocumentsView,
      redirect: '/documents/active',
      children: [
        { path: 'active', name: 'documents-active', component: DocumentsActiveSub },
        { path: 'processing', name: 'documents-processing', component: DocumentsProcessingSub },
        { path: 'uploads', name: 'documents-uploads', component: DocumentsUploadsSub },
        { path: 'deleted', name: 'documents-deleted', component: DocumentsDeletedSub },
      ],
    },

    { path: '/capabilities', name: 'capabilities', component: CapabilitiesView },

    {
      path: '/policies',
      name: 'policies',
      component: PoliciesView,
      redirect: '/policies/chat',
      children: [
        { path: 'chat', name: 'policies-chat', component: PoliciesChatSub },
        { path: 'background', name: 'policies-background', component: PoliciesBackgroundSub },
        { path: 'external', name: 'policies-external', component: PoliciesExternalSub },
      ],
    },

    { path: '/skills', name: 'skills', component: SkillsView },
    { path: '/mcp', name: 'mcp', component: McpView },
    { path: '/import-export', name: 'import-export', component: ImportExportView },

    // Catch-all → providers.
    { path: '/:pathMatch(.*)*', redirect: '/providers' },
  ],
});

// True if the gate hard-redirected on initial nav; main.ts reads it to skip the mount.
let _gateRedirected = false;
export function authGateRedirected(): boolean {
  return _gateRedirected;
}

function hardRedirect(to: string): false {
  _gateRedirected = true;
  window.location.replace(to);
  return false;
}

/**
 * Brain auth gate: !account → /on-boarding; !session → /login; !providers →
 * providersOnly LOCK (stay mounted, force Providers panel, NOT a hard redirect).
 * Network failure → stay; the real API endpoints guard themselves.
 */
router.beforeEach(async () => {
  const { useShellStore } = await import('./stores/shell');
  const shell = useShellStore();

  let status;
  try {
    status = await system.authStatus();
  } catch (err) {
    if (err instanceof HttpError || err instanceof AuthError) {
      // Server responded not-ok — treat as all-false → onboarding.
      return hardRedirect('/on-boarding/');
    }
    // Server unreachable — stay; real endpoints still guard.
    return true;
  }

  const { has_master_account, has_session, has_providers } = status;

  if (!has_master_account) {
    return hardRedirect('/on-boarding/');
  }
  if (!has_session) {
    return hardRedirect(
      '/login/?next=' + encodeURIComponent(window.location.pathname + window.location.search),
    );
  }

  if (!has_providers) {
    shell.providersOnly = true;
    if (router.currentRoute.value.name !== 'providers') {
      return { name: 'providers' };
    }
  }

  return true;
});
