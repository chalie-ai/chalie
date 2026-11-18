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
 * (`voice_state`) instead, so `stateFor` takes that as the fallback. The two
 * can disagree only when a frame was missed, and then the fetched value is the
 * newer one: the server writes a row only once synthesis has ended, so a
 * fetched ready/failed is final and outranks a live 'pending' that never heard
 * the ending.
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
   * Forget one row's live state, so it reads from its fetched snapshot again —
   * for a press whose request never got an answer from the server.
   */
  function forget(transcriptId: number): void {
    delete byTranscriptId.value[transcriptId];
  }

  /**
   * Forget every live 'pending'. Called when the socket drops: the frame that
   * would end it may have died with the connection and is never resent, and a
   * 'pending' nothing will end keeps the speaker button dead until a reload.
   * Forgotten, the button is pressable again, and a press asks the server.
   */
  function forgetPending(): void {
    for (const [id, state] of Object.entries(byTranscriptId.value)) {
      if (state === 'pending') delete byTranscriptId.value[Number(id)];
    }
  }

  /**
   * The state to paint: whatever arrived live, else the snapshot the message
   * was fetched with, except that a fetched final state beats a live
   * 'pending'. `null` means no attempt has been recorded at all — the button
   * renders normally and the first press starts the pipeline.
   */
  function stateFor(
    transcriptId: number,
    fetched: VoiceState | null = null,
  ): VoiceState | null {
    const live = byTranscriptId.value[transcriptId];
    if (live === 'pending' && fetched) return fetched;
    return live ?? fetched;
  }

  return { byTranscriptId, record, forget, forgetPending, stateFor };
});
