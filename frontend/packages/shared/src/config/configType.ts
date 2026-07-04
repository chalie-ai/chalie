/**
 * ConfigType — the ProcessorConfig identity the thread API routes on, mirroring the
 * backend `ConfigTypeEnum` (services/config_type.py). This is the ONLY surface the
 * frontend speaks: it never sees the transcript "channel" (a pure backend storage key
 * the API resolves type → config → channel server-side). JS and Python always match
 * on this `type` value — keep the two enums in lockstep.
 */
export const ConfigType = {
  USER: 'user',
  SCHEDULED: 'scheduled',
  DISCOVERY: 'discovery',
} as const;

export type ConfigType = (typeof ConfigType)[keyof typeof ConfigType];
