// @vitest-environment happy-dom
/**
 * The speaker button never stays dead for want of a frame or an answer.
 *
 * Its live state comes over the socket, and a frame the socket loses is never
 * resent; a press's request can also fail without the server ever ruling on the
 * speech. In both cases the button used to stay disabled until a reload.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { mount } from '@vue/test-utils';
import { createPinia, setActivePinia } from 'pinia';
import BubbleFooter from './BubbleFooter.vue';
import { on } from '../../composables/useEventBus';
import { useVoiceTranscriptsStore } from '../../stores/voiceTranscripts';
import type { ConversationMessage } from '../../api/conversation';

const { transcriptMock } = vi.hoisted(() => ({ transcriptMock: vi.fn() }));
vi.mock('../../api/voice', () => ({
  voice: { transcript: (id: number) => transcriptMock(id) },
}));

const SPEAK_BUTTON = 'button.speech-form__act-btn--speak';
const ROW = 58;

function settledReply(voiceState: ConversationMessage['voice_state']): ConversationMessage {
  return {
    id: String(ROW),
    role: 'assistant',
    content: '<p>Malta is the easiest quiet winter.</p>',
    timestamp: '21:06',
    turn_id: 7,
    settled: true,
    voice_state: voiceState,
  } as ConversationMessage;
}

describe('BubbleFooter speaker button', () => {
  const spoken: number[] = [];
  let unbind: () => void = () => {};

  beforeEach(() => {
    setActivePinia(createPinia());
    transcriptMock.mockReset();
    spoken.length = 0;
    unbind = on('chalie:speak-message', ({ transcriptId }) => spoken.push(transcriptId));
  });

  afterEach(() => unbind());

  it('plays a reply the server has ready even though the live state is still waiting on a frame it missed', async () => {
    useVoiceTranscriptsStore().record(ROW, 'pending');
    const wrapper = mount(BubbleFooter, { props: { message: settledReply('ready') } });

    const button = wrapper.find(SPEAK_BUTTON);
    expect(button.attributes('disabled')).toBeUndefined();
    await button.trigger('click');

    expect(spoken).toEqual([ROW]);
  });

  it('stays pressable after a press whose request never reached the server, and the next press plays', async () => {
    transcriptMock.mockRejectedValueOnce(new TypeError('Failed to fetch'));
    vi.spyOn(console, 'error').mockImplementation(() => {});
    const wrapper = mount(BubbleFooter, { props: { message: settledReply(null) } });

    await wrapper.find(SPEAK_BUTTON).trigger('click');
    await vi.waitFor(() => expect(transcriptMock).toHaveBeenCalledTimes(1));
    await wrapper.vm.$nextTick();

    const button = wrapper.find(SPEAK_BUTTON);
    expect(button.classes()).not.toContain('speech-form__act-btn--failed');
    expect(button.attributes('disabled')).toBeUndefined();

    transcriptMock.mockResolvedValueOnce(new Response('audio', { status: 200 }));
    await button.trigger('click');
    await vi.waitFor(() => expect(spoken).toEqual([ROW]));
  });

  it('still goes dead when the server itself says the speech could not be made', async () => {
    transcriptMock.mockResolvedValueOnce(new Response(null, { status: 409 }));
    const wrapper = mount(BubbleFooter, { props: { message: settledReply(null) } });

    await wrapper.find(SPEAK_BUTTON).trigger('click');
    await vi.waitFor(() => expect(wrapper.find(SPEAK_BUTTON).classes()).toContain('speech-form__act-btn--failed'));
    expect(wrapper.find(SPEAK_BUTTON).attributes('disabled')).toBeDefined();
  });
});
