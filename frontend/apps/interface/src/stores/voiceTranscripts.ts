/**
 * Voice-playback state, keyed by transcript row id.
 *
 * Speech is synthesized in the background the moment a reply settles, so the
 * speaker button has to paint a state it never initiated. Two writers feed it:
 * the `voice_transcript` WS frame (`utils/driftDispatcher.ts`), pushed as the
 * pipeline moves pending → ready/failed, and the lazy start a press performs
 * on history that predates the feature.
 *
 * This store holds only what has arrived over the wire THIS session. A reload
 * empties it and the state comes back on each message from the API projection
 * (`voice_state`) instead — so `stateFor` takes that as the fallback and the
 * two never fight: a live frame always wins over the fetched snapshot it
 * supersedes.
 */
import { ref } from 'vue';
import { defineStore } from 'pinia';

export type VoiceState = 'pending' | 'ready' | 'failed';

export const useVoiceTranscriptsStore = defineStore('voiceTranscripts', () => {
  const byTranscriptId = ref<Record<number, VoiceState>>({});

  /** Record a pipeline state for one transcript row. */
  function record(transcriptId: number, state: VoiceState): void {
    byTranscriptId.value[transcriptId] = state;
  }

  /**
   * The state to paint: whatever arrived live, else the snapshot the message
   * was fetched with. `null` means no attempt has been recorded at all — the
   * button renders normally and the first press starts the pipeline.
   */
  function stateFor(
    transcriptId: number,
    fetched: VoiceState | null = null,
  ): VoiceState | null {
    return byTranscriptId.value[transcriptId] ?? fetched;
  }

  return { byTranscriptId, record, stateFor };
});
