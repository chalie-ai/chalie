/**
 * Rich-media payload types (Task B0, P1b).
 *
 * Single import surface for every card payload interface. Each interface is
 * authored co-located in its card SFC (so the type lives next to the code that
 * reads it); this barrel re-exports them for consumers that want the payload
 * shapes without importing a whole `.vue` component.
 */
export type { WeatherPayload } from './WeatherCard.vue';
export type { SchedulerPayload } from './SchedulerCard.vue';
export type { ListPayload } from './ListCard.vue';
export type { TimerPayload } from './TimerCard.vue';
export type { CalendarPayload } from './CalendarCard.vue';
export type { ContactsPayload } from './ContactsCard.vue';
