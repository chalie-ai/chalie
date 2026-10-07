import type { FunctionalComponent } from 'vue';
import {
  BookOpen,
  Brain,
  Calendar,
  DatabaseBackup,
  LayoutGrid,
  List,
  Network,
  Server,
  Settings,
  ShieldCheck,
} from '@lucide/vue';

export interface SubItem {
  id: string;
  label: string;
}

export interface NavItem {
  id: string;
  label: string;
  icon: FunctionalComponent;
  group: 'cognition' | 'system';
  sub?: SubItem[];
}

export const NAV: NavItem[] = [
  { id: 'providers', label: 'Providers', icon: LayoutGrid, group: 'cognition' },
  {
    id: 'cognition',
    label: 'Cognition',
    icon: Brain,
    group: 'cognition',
    sub: [
      { id: 'memory', label: 'Memory' },
      { id: 'tools', label: 'Tools' },
      { id: 'world', label: 'World state' },
      { id: 'personality', label: 'Personality' },
      { id: 'errors', label: 'Errors' },
      { id: 'usage', label: 'Usage' },
      { id: 'compaction', label: 'Compacted Summary' },
    ],
  },
  { id: 'scheduler', label: 'Scheduler', icon: Calendar, group: 'cognition' },
  { id: 'lists', label: 'Lists', icon: List, group: 'cognition' },
  { id: 'capabilities', label: 'Capabilities', icon: Settings, group: 'system' },
  {
    id: 'policies',
    label: 'Policies',
    icon: ShieldCheck,
    group: 'system',
    sub: [
      { id: 'chat', label: 'Chat' },
      { id: 'background', label: 'Background' },
      { id: 'external', label: 'External agent' },
    ],
  },
  { id: 'skills', label: 'Skills', icon: BookOpen, group: 'system' },
  { id: 'mcp', label: 'MCP', icon: Server, group: 'system' },
  { id: 'import-export', label: 'Import / Export', icon: DatabaseBackup, group: 'system' },
  { id: 'system', label: 'System', icon: Network, group: 'system' },
];
