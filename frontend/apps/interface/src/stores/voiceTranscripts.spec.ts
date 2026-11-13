/**
 * A speech state is never stuck on 'pending' by a frame the socket lost.
 *
 * Live frames normally outrank the snapshot a message was fetched with, but
 * 'pending' only waits for a later frame — when the fetch already holds the
 * verdict, the verdict wins.
 */
import { beforeEach, describe, expect, it } from 'vitest';
import { createPinia, setActivePinia } from 'pinia';
import { useVoiceTranscriptsStore } from './voiceTranscripts';

describe('voice transcript state', () => {
  beforeEach(() => setActivePinia(createPinia()));

  it('takes the fetched verdict over a live pending that never heard its outcome', () => {
    const voice = useVoiceTranscriptsStore();
    voice.record(58, 'pending');

    expect(voice.stateFor(58, 'ready')).toBe('ready');
    expect(voice.stateFor(58, 'failed')).toBe('failed');
    expect(voice.stateFor(58, null)).toBe('pending');
  });

  it('takes a live verdict over an older fetched one', () => {
    const voice = useVoiceTranscriptsStore();
    voice.record(58, 'ready');
    voice.record(59, 'failed');

    expect(voice.stateFor(58, 'failed')).toBe('ready');
    expect(voice.stateFor(59, null)).toBe('failed');
    expect(voice.stateFor(60, 'ready')).toBe('ready');
  });

  it('forgets only what is still being prepared', () => {
    const voice = useVoiceTranscriptsStore();
    voice.record(58, 'pending');
    voice.record(59, 'ready');
    voice.record(60, 'pending');
    voice.record(61, 'failed');

    voice.forgetPending();

    expect([58, 59, 60, 61].map((id) => voice.stateFor(id))).toEqual([null, 'ready', null, 'failed']);
  });
});
