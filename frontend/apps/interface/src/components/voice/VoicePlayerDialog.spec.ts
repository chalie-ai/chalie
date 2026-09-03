// @vitest-environment happy-dom
/**
 * VoicePlayerDialog — feature spec for surviving a screen lock mid-playback.
 *
 * Real DOM (happy-dom), the REAL VoicePlayerDialog.vue mounted with
 * `@vue/test-utils`'s `mount`, the REAL `webPlatformAdapter`, the REAL event
 * bus and the REAL `api/voice.ts` fetch call. Nothing of ours is mocked.
 *
 * The two stand-ins are browser platform surfaces happy-dom does not implement:
 * `globalThis.AudioContext` and `globalThis.fetch`. The AudioContext stand-in
 * models WebKit's FOUR-value state machine — WebKit ships
 * `enum AudioContextState { "suspended", "running", "interrupted", "closed" }`
 * (Source/WebCore/Modules/webaudio/AudioContextState.idl), where `interrupted`
 * is what an iOS screen lock produces. The Web Audio draft's own note names the
 * trigger: "a statechange event is not fired ... e.g. when a phone call comes
 * in or when the screen gets locked."
 *
 * Sound only comes out of a context that is actually `running`, so every
 * assertion here is "a buffer source was started WHILE the context was
 * running", never merely "start() was called".
 */
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { mount } from '@vue/test-utils';
import type { VueWrapper } from '@vue/test-utils';
import VoicePlayerDialog from './VoicePlayerDialog.vue';
import { emit } from '../../composables/useEventBus';

type CtxState = 'suspended' | 'running' | 'interrupted' | 'closed';

const FAKE_BUFFER = { duration: 12, length: 288_000, sampleRate: 24_000, numberOfChannels: 1 };

class FakeBufferSource {
  buffer: unknown = null;
  onended: (() => void) | null = null;
  /** The context's state at the moment start() ran — 'running' means audible. */
  ctxStateAtStart: CtxState | null = null;
  startWhen: number | null = null;
  startOffset: number | null = null;

  constructor(private readonly ctx: FakeAudioContext) {}

  connect(): void {}
  disconnect(): void {}
  stop(): void {}

  /** Per spec, start() on a non-running context queues rather than throwing. */
  start(when = 0, offset = 0): void {
    this.ctxStateAtStart = this.ctx.state;
    this.startWhen = when;
    this.startOffset = offset;
  }
}

class FakeAudioContext extends EventTarget {
  state: CtxState = 'suspended';
  currentTime = 0;
  destination = {};
  readonly sources: FakeBufferSource[] = [];
  resumeCalls = 0;
  closed = false;
  decodedBytes = 0;
  decodeFails = false;
  /**
   * When false, resume() leaves the context stuck — the real WebKit behaviour
   * reported in bugs 273511 ("stuck on 'interrupted' ... .resume() does not
   * work") and 263627, where only a fresh context recovers.
   */
  resumeWorks = true;

  createBufferSource(): FakeBufferSource {
    const source = new FakeBufferSource(this);
    this.sources.push(source);
    return source;
  }

  /** Callback form — the player targets it for iOS compatibility. */
  decodeAudioData(
    data: ArrayBuffer,
    onSuccess: (b: unknown) => void,
    onError: (e: unknown) => void,
  ): void {
    this.decodedBytes = data.byteLength;
    if (this.decodeFails) onError(new Error('decode failed'));
    else onSuccess(FAKE_BUFFER);
  }

  async resume(): Promise<void> {
    this.resumeCalls += 1;
    if (this.resumeWorks) this.#transition('running');
  }

  async close(): Promise<void> {
    this.closed = true;
    this.#transition('closed');
  }

  /** The OS takes the audio session — what an iOS screen lock does. */
  interrupt(): void {
    this.#transition('interrupted');
  }

  #transition(next: CtxState): void {
    if (this.state === next) return;
    this.state = next;
    this.dispatchEvent(new Event('statechange'));
  }
}

let contexts: FakeAudioContext[] = [];

/** The source a context actually made audible, if any. */
function audibleSources(ctx: FakeAudioContext): FakeBufferSource[] {
  return ctx.sources.filter((s) => s.ctxStateAtStart === 'running');
}

/** Drain the component's promise chain (fetch -> arrayBuffer -> decode -> play). */
async function settle(wrapper: VueWrapper): Promise<void> {
  for (let i = 0; i < 12; i++) await Promise.resolve();
  await wrapper.vm.$nextTick();
}

/** Mount the player and drive it through a real open, ending mid-playback. */
async function openAndPlay(): Promise<{ wrapper: VueWrapper; ctx: FakeAudioContext }> {
  const wrapper = mount(VoicePlayerDialog, { attachTo: document.body });
  emit('chalie:speak-message', { transcriptId: 42 });
  await settle(wrapper);

  const ctx = contexts[0];
  expect(ctx, 'the player should have built an AudioContext').toBeTruthy();
  expect(ctx.decodedBytes, 'the fetched audio should have been decoded').toBe(16);
  expect(audibleSources(ctx), 'playback should be audible before the lock').toHaveLength(1);
  expect(playLabel(wrapper)).toBe('Pause');
  return { wrapper, ctx };
}

function playLabel(wrapper: VueWrapper): string | undefined {
  return wrapper.find('.voice-player__btn--play').attributes('aria-label');
}

beforeEach(() => {
  contexts = [];
  (globalThis as unknown as { AudioContext: unknown }).AudioContext = function () {
    const ctx = new FakeAudioContext();
    contexts.push(ctx);
    return ctx;
  };
  (globalThis as unknown as { fetch: unknown }).fetch = async () => ({
    ok: true,
    status: 200,
    arrayBuffer: async () => new ArrayBuffer(16),
  });
});

afterEach(() => {
  document.body.innerHTML = '';
});

describe('VoicePlayerDialog — the screen locks mid-playback', () => {
  it('parks playback at its position instead of claiming it is still playing', async () => {
    const { wrapper, ctx } = await openAndPlay();

    // 7 seconds in, the phone screen locks: WebKit moves running -> interrupted.
    ctx.currentTime = 7;
    ctx.interrupt();
    await wrapper.vm.$nextTick();

    // The button must offer Play, not Pause — otherwise the user's first tap
    // pauses an already-dead stream and the player looks broken.
    expect(playLabel(wrapper)).toBe('Play');

    wrapper.unmount();
  });

  it('resumes from where it stopped when the user taps play, with no page reload', async () => {
    const { wrapper, ctx } = await openAndPlay();

    ctx.currentTime = 7;
    ctx.interrupt();
    await wrapper.vm.$nextTick();

    const resumesBefore = ctx.resumeCalls;
    await wrapper.find('.voice-player__btn--play').trigger('click');
    await settle(wrapper);

    // resume() must have been called even though the state was 'interrupted'
    // and never 'suspended' — the old guard tested only for 'suspended'.
    expect(ctx.resumeCalls).toBeGreaterThan(resumesBefore);
    expect(ctx.state).toBe('running');

    // A second source actually became audible, carrying on from 7s.
    const audible = audibleSources(ctx);
    expect(audible).toHaveLength(2);
    expect(audible[1].startOffset).toBeCloseTo(7, 5);
    expect(playLabel(wrapper)).toBe('Pause');

    // No new context was needed — the existing one recovered.
    expect(contexts).toHaveLength(1);

    wrapper.unmount();
  });

  it('rebuilds the audio context when the interrupted one refuses to resume', async () => {
    const { wrapper, ctx } = await openAndPlay();

    // WebKit bug 273511: the context is stuck on 'interrupted' and resume()
    // does nothing. Today the only cure is reloading the page.
    ctx.resumeWorks = false;
    ctx.currentTime = 7;
    ctx.interrupt();
    await wrapper.vm.$nextTick();

    await wrapper.find('.voice-player__btn--play').trigger('click');
    await settle(wrapper);

    // The dead context was torn down and a fresh one built in its place.
    expect(ctx.resumeCalls).toBeGreaterThan(0);
    expect(ctx.closed).toBe(true);
    expect(contexts).toHaveLength(2);

    // Audio really resumed on the replacement, from where it left off.
    const revived = contexts[1];
    expect(revived.state).toBe('running');
    const audible = audibleSources(revived);
    expect(audible).toHaveLength(1);
    expect(audible[0].startOffset).toBeCloseTo(7, 5);
    expect(playLabel(wrapper)).toBe('Pause');

    wrapper.unmount();
  });
});
