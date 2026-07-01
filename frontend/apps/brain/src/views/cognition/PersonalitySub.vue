<script setup lang="ts">
import { onMounted, reactive, ref } from 'vue';
import { cognition } from '../../api/cognition';
import { useToast } from '../../composables/useToast';
import { HttpError } from '@chalie/shared';
import EmptyState from '../../ui/EmptyState.vue';

const { show: showToast } = useToast();

const loading = ref(true);
const loadFailed = ref(false);

const personality = reactive<{
  warmth: number;
  mood: number;
  expressiveness: number;
  curiosity: number;
  humor: number;
}>({ warmth: 0, mood: 0, expressiveness: 0, curiosity: 0, humor: 0 });

const sliders = [
  { key: 'warmth' as const, label: 'Warmth', left: 'Cool', right: 'Warm' },
  { key: 'mood' as const, label: 'Mood', left: 'Level-headed', right: 'Moody' },
  { key: 'expressiveness' as const, label: 'Expressiveness', left: 'Reserved', right: 'Vocal' },
  { key: 'curiosity' as const, label: 'Curiosity', left: 'Matter-of-fact', right: 'Inquisitive' },
  { key: 'humor' as const, label: 'Humor', left: 'Dry', right: 'Playful' },
];

onMounted(async () => {
  try {
    const data = await cognition.personality();
    const t = data.tuple || [0, 0, 0, 0, 0];
    personality.warmth = t[0];
    personality.mood = t[1];
    personality.expressiveness = t[2];
    personality.curiosity = t[3];
    personality.humor = t[4];
  } catch {
    loadFailed.value = true;
  } finally {
    loading.value = false;
  }
});

async function save() {
  const tuple: [number, number, number, number, number] = [
    personality.warmth,
    personality.mood,
    personality.expressiveness,
    personality.curiosity,
    personality.humor,
  ];
  try {
    await cognition.setPersonality({ tuple });
    showToast('Personality saved', 'success');
  } catch (e: unknown) {
    showToast(e instanceof HttpError ? 'Failed to save' : 'Network error', 'error');
  }
}
</script>

<template>
  <template v-if="loading"><div class="loading">Loading…</div></template>
  <template v-else-if="loadFailed"><EmptyState message="Failed to load data." /></template>
  <template v-else>
    <p class="panel-desc">
      Adjust how Chalie communicates. Changes take effect on the next message.
    </p>
    <div class="personality-grid">
      <div v-for="s in sliders" :key="s.key" class="personality-field">
        <label class="personality-label">{{ s.label }}</label>
        <div class="personality-row">
          <span class="pole pole-left">{{ s.left }}</span>
          <input
            v-model.number="personality[s.key]"
            type="range"
            min="-2"
            max="2"
            step="1"
            class="personality-range"
            :data-key="s.key"
          />
          <span class="pole pole-right">{{ s.right }}</span>
        </div>
      </div>
    </div>
    <div class="personality-actions">
      <button class="btn btn-primary" @click="save">Save</button>
    </div>
  </template>
</template>
