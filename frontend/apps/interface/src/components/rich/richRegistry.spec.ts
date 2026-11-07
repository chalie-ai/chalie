import { describe, expect, it } from 'vitest';
import { preloadRichCards, resolveRichCard } from './richRegistry';

describe('preloadRichCards', () => {
  it('resolves a preloaded card to its loaded component, so it mounts in the same render as its turn', async () => {
    const { default: WeatherCard } = await import('./WeatherCard.vue');
    expect(resolveRichCard('weather_1')?.component).not.toBe(WeatherCard);

    await preloadRichCards([{ type: 'rich', tag: 'weather_1' }]);

    expect(resolveRichCard('weather_2')?.component).toBe(WeatherCard);
  });
});
