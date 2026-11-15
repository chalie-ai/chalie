/**
 * toolCatalog — what each kind of tool call is called in a step's summary:
 * its icon, its phrase for one and for {n}, and the argument naming what it
 * acts on, so calls on the same value count once. A kind is a tool, or a tool
 * and its `action` argument. `plan` is hidden — the plan card shows it. A kind
 * also names the view its row opens to; a kind without one opens to its
 * arguments.
 */
import {
  ArrowDownUp, ArrowLeft, BookmarkPlus, BookOpen, BookOpenText, Blocks, Bot, Brain, BrainCog,
  Calendar, CalendarClock, CalendarCog, CalendarPlus, CalendarSearch, CalendarX, Camera, CloudSun, Contact,
  Eraser, Eye, FileCode, FileDown, FilePenLine, FilePlus, FileSearch, FileText, FolderInput, FolderPlus, Forward,
  Globe, Hand, History, House, Image, Images, ListChecks, ListTodo, Lock, Mail, MailCheck, MailOpen, MessagesSquare,
  MonitorSmartphone, MousePointerClick, Newspaper, NotebookPen, PackageSearch, Palette, Paperclip, Plug, Reply,
  Router, Search, Send, Sparkles, SquarePen, SquareTerminal, TextCursorInput, TextSearch, Timer, Trash2,
  Unplug, UserRoundPen, Wrench,
} from '@lucide/vue';
import type { LucideIcon } from '@lucide/vue';
import type { StepCall } from './turnSteps';

/** How an opened row shows its call (see toolDetail). */
export type CallView = 'command' | 'edit' | 'write' | 'read' | 'browser' | 'browse' | 'delegate' | 'email';

export interface CallKind {
  /** The group a call counts in: its tool, or `tool.action`. */
  key: string;
  icon: LucideIcon;
  /** The lower-case phrase for one, then for `{n}`. */
  one: string;
  many: string;
  /** The argument naming what the call acts on; calls on one value count once. */
  target?: string;
  hidden?: boolean;
  view?: CallView;
  /** The arguments a `browser` view shows. */
  fields?: string[];
}

const kind = (key: string, icon: LucideIcon, one: string, many: string, target?: string): CallKind => ({
  key, icon, one, many, target,
});

/** A kind that counts its calls: "checked memory 3 times". */
const times = (key: string, icon: LucideIcon, phrase: string): CallKind => kind(key, icon, phrase, `${phrase} {n} times`);

const viewed = (view: CallView, kinds: CallKind[]): CallKind[] => kinds.map((k) => ({ ...k, view }));

const showing = (k: CallKind, ...fields: string[]): CallKind => ({ ...k, fields });

const skillKinds = (tool: string): CallKind[] => [
  kind(`${tool}.create`, NotebookPen, 'created 1 skill', 'created {n} skills', 'title'),
  kind(`${tool}.edit`, NotebookPen, 'edited 1 skill', 'edited {n} skills', 'title'),
  kind(`${tool}.delete`, NotebookPen, 'deleted 1 skill', 'deleted {n} skills', 'title'),
  kind(`${tool}.read`, NotebookPen, 'read 1 skill', 'read {n} skills', 'title'),
  times(`${tool}.list`, NotebookPen, 'listed skills'),
];

const KINDS: CallKind[] = [
  { ...times('plan', ListTodo, 'updated the plan'), hidden: true },
  { ...kind('bash', SquareTerminal, 'ran 1 command', 'ran {n} commands'), view: 'command' },
  { ...kind('write_file', FilePlus, 'wrote 1 file', 'wrote {n} files', 'path'), view: 'write' },
  { ...kind('edit_file', FilePenLine, 'edited 1 file', 'edited {n} files', 'path'), view: 'edit' },
  { ...kind('read', FileText, 'read 1 file', 'read {n} files', 'source'), view: 'read' },
  kind('delete', Trash2, 'deleted 1 item', 'deleted {n} items', 'path'),
  kind('make_dir', FolderPlus, 'created 1 folder', 'created {n} folders', 'path'),
  kind('move', FolderInput, 'moved 1 item', 'moved {n} items', 'current_path'),
  kind('set_permissions', Lock, 'changed permissions on 1 item', 'changed permissions on {n} items', 'path'),
  kind('run_script', FileCode, 'ran 1 script', 'ran {n} scripts', 'path'),
  kind('image_preview', Image, 'showed 1 image', 'showed {n} images', 'file_path'),
  kind('vision', Eye, 'looked at 1 image', 'looked at {n} images', 'image'),
  kind('web_fetch', FileDown, 'fetched 1 page', 'fetched {n} pages', 'url'),
  times('search_files', FileSearch, 'searched files'),
  times('search', Search, 'searched the web'),
  times('news', Newspaper, 'searched the news'),
  times('image_search', Images, 'searched for images'),
  times('programming_docs_search', BookOpenText, 'searched programming docs'),
  times('chalie_docs', BookOpen, "looked up Chalie's docs"),
  times('recall', Brain, 'checked memory'),
  times('review_tool_calls', History, 'reviewed earlier tool calls'),
  times('review_transcript', MessagesSquare, 'reviewed the conversation'),
  times('find_skills', Sparkles, 'looked for a skill'),
  times('find_tools', PackageSearch, 'looked for tools'),
  times('weather', CloudSun, 'checked the weather'),
  times('user_device', MonitorSmartphone, 'checked your device'),
  kind('timer', Timer, 'started 1 timer', 'started {n} timers'),
  kind('save_graph', BrainCog, 'saved 1 fact', 'saved {n} facts', 'subject'),
  kind('delete_graph', Eraser, 'forgot 1 fact', 'forgot {n} facts', 'subject'),
  kind('save_map', BookmarkPlus, 'saved 1 memory', 'saved {n} memories'),
  times('save_synthesis', UserRoundPen, 'updated the user profile'),
  ...viewed('delegate', [
    kind('code_agent', Bot, 'delegated 1 coding task', 'delegated {n} coding tasks'),
    kind('web_search', Bot, 'delegated 1 web search', 'delegated {n} web searches'),
    kind('pim', Bot, 'delegated 1 mail & calendar task', 'delegated {n} mail & calendar tasks'),
    kind('subtask.start', Bot, 'started 1 subtask', 'started {n} subtasks'),
    kind('subtask.message', Bot, 'messaged 1 subtask', 'messaged {n} subtasks', 'subtask'),
  ]),
  { ...kind('web_browse', Bot, 'delegated 1 browsing task', 'delegated {n} browsing tasks'), view: 'browse' },
  ...viewed('browser', [
    showing(kind('browser.open', Globe, 'opened 1 page', 'opened {n} pages', 'url'), 'url'),
    showing(kind('browser.read', FileText, 'read 1 page', 'read {n} pages'), 'section'),
    showing(times('browser.find', TextSearch, 'searched a page'), 'query'),
    showing(kind('browser.click', MousePointerClick, 'clicked 1 element', 'clicked {n} elements', 'target'), 'target'),
    showing(kind('browser.fill', TextCursorInput, 'filled 1 field', 'filled {n} fields', 'target'), 'target', 'value'),
    showing(kind('browser.select', ListChecks, 'chose 1 option', 'chose {n} options', 'target'), 'target', 'value'),
    times('browser.scroll', ArrowDownUp, 'scrolled'),
    times('browser.back', ArrowLeft, 'went back'),
    kind('browser.screenshot', Camera, 'took 1 screenshot', 'took {n} screenshots'),
    showing(kind('browser.style', Palette, "inspected 1 element's style", "inspected {n} elements' styles", 'target'), 'target'),
    showing(kind('browser.handoff', Hand, 'handed 1 page to you', 'handed {n} pages to you'), 'reason'),
  ]),
  times('calendar.list_events', CalendarSearch, 'checked the calendar'),
  kind('calendar.get_event', Calendar, 'read 1 event', 'read {n} events', 'uid'),
  kind('calendar.create_event', CalendarPlus, 'created 1 event', 'created {n} events'),
  kind('calendar.update_event', CalendarCog, 'updated 1 event', 'updated {n} events', 'uid'),
  kind('calendar.delete_event', CalendarX, 'deleted 1 event', 'deleted {n} events', 'uid'),
  times('contacts.list', Contact, 'searched contacts'),
  kind('contacts.get', Contact, 'looked up 1 contact', 'looked up {n} contacts', 'identifier'),
  ...viewed('email', [
    times('email.search', Mail, 'searched email'),
    kind('email.read', MailOpen, 'read 1 email', 'read {n} emails', 'uid'),
    kind('email.draft', SquarePen, 'drafted 1 email', 'drafted {n} emails'),
    kind('email.send', Send, 'sent 1 email', 'sent {n} emails'),
    kind('email.reply', Reply, 'replied to 1 email', 'replied to {n} emails', 'uid'),
    kind('email.forward', Forward, 'forwarded 1 email', 'forwarded {n} emails', 'uid'),
    kind('email.download', Paperclip, 'downloaded attachments from 1 email', 'downloaded attachments from {n} emails', 'uid'),
    kind('email.manage', MailCheck, 'managed 1 email', 'managed {n} emails', 'uid'),
  ]),
  times('home.list_devices', House, 'listed devices'),
  kind('home.get_state', House, 'checked 1 device', 'checked {n} devices', 'entity_id'),
  kind('home.control', House, 'controlled 1 device', 'controlled {n} devices', 'entity_id'),
  times('home.list_automations', House, 'listed automations'),
  kind('home.trigger_automation', House, 'ran 1 automation', 'ran {n} automations', 'automation_id'),
  times('home.subscribe_events', House, 'watched home events'),
  times('list.list_all', ListChecks, 'listed your lists'),
  kind('list.create', ListChecks, 'created 1 list', 'created {n} lists', 'name'),
  kind('list.view', ListChecks, 'viewed 1 list', 'viewed {n} lists', 'list'),
  kind('list.add', ListChecks, 'added 1 item', 'added {n} items', 'items'),
  kind('list.check', ListChecks, 'checked off 1 item', 'checked off {n} items', 'items'),
  kind('list.remove', ListChecks, 'removed 1 item', 'removed {n} items', 'items'),
  kind('list.clear', ListChecks, 'cleared 1 list', 'cleared {n} lists', 'list'),
  kind('list.rename', ListChecks, 'renamed 1 list', 'renamed {n} lists', 'list'),
  kind('list.delete', ListChecks, 'deleted 1 list', 'deleted {n} lists', 'list'),
  kind('schedule.create', CalendarClock, 'scheduled 1 task', 'scheduled {n} tasks'),
  times('schedule.list', CalendarClock, 'listed schedules'),
  times('schedule.search', CalendarClock, 'searched schedules'),
  kind('schedule.cancel', CalendarClock, 'cancelled 1 schedule', 'cancelled {n} schedules', 'item_id'),
  kind('schedule.update', CalendarClock, 'updated 1 schedule', 'updated {n} schedules', 'item_id'),
  kind('schedule.enable', CalendarClock, 'turned on 1 schedule', 'turned on {n} schedules', 'item_id'),
  kind('schedule.disable', CalendarClock, 'paused 1 schedule', 'paused {n} schedules', 'item_id'),
  ...skillKinds('skill_builder'),
  ...skillKinds('skill_manager'),
  times('mcp_manager.list', Plug, 'listed MCP servers'),
  kind('mcp_manager.add', Plug, 'connected 1 server', 'connected {n} servers', 'name'),
  kind('mcp_manager.enable', Plug, 'enabled 1 server', 'enabled {n} servers', 'name'),
  kind('mcp_manager.disable', Unplug, 'disabled 1 server', 'disabled {n} servers', 'name'),
  times('mcp_tools.list', Blocks, 'listed MCP tools'),
  kind('mcp_tools.activate', Blocks, 'turned on 1 tool', 'turned on {n} tools', 'tools'),
  times('ubiquiti.list_devices', Router, 'listed network devices'),
  times('ubiquiti.list_clients', Router, 'listed clients'),
  times('ubiquiti.list_wifi', Router, 'listed Wi-Fi networks'),
  times('ubiquiti.list_port_forwards', Router, 'listed port forwards'),
  times('ubiquiti.list_traffic_rules', Router, 'listed traffic rules'),
  times('ubiquiti.site_health', Router, 'checked network health'),
  kind('ubiquiti.device_status', Router, 'checked 1 device', 'checked {n} devices', 'target'),
  kind('ubiquiti.block_client', Router, 'blocked 1 client', 'blocked {n} clients', 'target'),
  kind('ubiquiti.unblock_client', Router, 'unblocked 1 client', 'unblocked {n} clients', 'target'),
  kind('ubiquiti.disconnect_client', Router, 'disconnected 1 client', 'disconnected {n} clients', 'target'),
  kind('ubiquiti.restart_device', Router, 'restarted 1 device', 'restarted {n} devices', 'target'),
  kind('ubiquiti.locate_device', Router, "flashed 1 device's locate light", "flashed {n} devices' locate lights", 'target'),
  kind('ubiquiti.stop_locate_device', Router, 'stopped locating 1 device', 'stopped locating {n} devices', 'target'),
  kind('ubiquiti.power_cycle_port', Router, 'power-cycled 1 port', 'power-cycled {n} ports'),
  kind('ubiquiti.update_wifi', Router, 'updated 1 Wi-Fi network', 'updated {n} Wi-Fi networks', 'wlan_id'),
  kind('ubiquiti.create_port_forward', Router, 'created 1 port forward', 'created {n} port forwards'),
  kind('ubiquiti.update_port_forward', Router, 'updated 1 port forward', 'updated {n} port forwards', 'rule_id'),
  kind('ubiquiti.update_traffic_rule', Router, 'updated 1 traffic rule', 'updated {n} traffic rules', 'rule_id'),
  kind('ubiquiti.authorize_guest', Router, 'authorized 1 guest', 'authorized {n} guests', 'target'),
];

const CATALOG = new Map(KINDS.map((k) => [k.key, k]));

const MCP = kind('_mcp', Plug, 'used 1 MCP tool', 'used {n} MCP tools');

/** A tool the catalog does not know: any MCP proxy (`_mcp_<server>_<tool>`)
 *  as one kind, else the tool by its name. */
function unlistedKind(tool: string): CallKind {
  return tool.startsWith('_mcp_') ? MCP : kind(tool, Wrench, `used ${tool}`, `used ${tool} {n} times`);
}

export function callKind(call: Pick<StepCall, 'tool_name' | 'params'>): CallKind {
  const action = call.params?.action;
  return CATALOG.get(`${call.tool_name}.${String(action)}`) ?? CATALOG.get(call.tool_name) ?? unlistedKind(call.tool_name);
}
