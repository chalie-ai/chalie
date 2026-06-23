/**
 * Barrel re-exporting every card payload interface (each authored in its SFC),
 * so consumers can import the shapes without pulling in a whole `.vue` component.
 */
export type { WeatherPayload } from './WeatherCard.vue';
export type { SchedulerPayload } from './SchedulerCard.vue';
export type { ListPayload } from './ListCard.vue';
export type { TimerPayload } from './TimerCard.vue';
export type { CalendarPayload } from './CalendarCard.vue';
export type { ContactsPayload } from './ContactsCard.vue';
