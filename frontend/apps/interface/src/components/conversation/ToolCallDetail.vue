<!-- An opened tool-call row: only what matters for its kind of call, read on
     its own the first time the row opens and again once a running call
     settles. A call that spawned a delegate turn ends on the way into that
     turn's transcript. -->
<script setup lang="ts">
import { computed, shallowRef, watch } from 'vue';
import { delegatePillAttrs } from '../../composables/useDelegatePill';
import { lastDetail, readDetail } from '../../utils/toolCallDetails';
import { toolDetailView, type DiffLine } from '../../utils/toolDetail';
import type { StepCall } from '../../utils/turnSteps';

const props = defineProps<{ call: StepCall }>();

const MARK: Record<DiffLine['op'], string> = { same: ' ', add: '+', remove: '-' };

const detail = shallowRef(lastDetail(props.call.id));
const failed = shallowRef(false);
const view = computed(() => (detail.value ? toolDetailView(detail.value) : null));
const delegate = computed(() => detail.value?.delegate ?? props.call.delegate);

watch(
  () => props.call.state,
  (state) => {
    failed.value = false;
    readDetail(props.call).then(
      (read) => {
        if (props.call.state === state) detail.value = read;
      },
      (err: unknown) => {
        console.error(`Couldn't load tool call ${props.call.id}`, err);
        if (props.call.state === state) failed.value = true;
      },
    );
  },
  { immediate: true },
);
</script>

<template>
  <div class="call-detail">
    <p v-if="failed" class="call-detail__note call-detail__note--error">Couldn't load this call.</p>
    <p v-else-if="!view" class="call-detail__note">Loading…</p>
    <template v-else>
      <p v-if="'heading' in view && view.heading" class="call-detail__heading">{{ view.heading }}</p>
      <pre v-if="view.kind === 'command'" class="call-detail__box call-detail__code">{{ view.command }}</pre>
      <template v-else-if="view.kind === 'read'">
        <p class="call-detail__text">{{ view.source }}</p>
        <p v-if="view.lines" class="call-detail__label">{{ view.lines }}</p>
      </template>
      <template v-else-if="view.kind === 'edit' || view.kind === 'write'">
        <p class="call-detail__text">{{ view.path }}</p>
        <div v-if="view.diff.length" class="call-detail__box call-detail__diff">
          <div v-for="(line, i) in view.diff" :key="i" class="diff-line" :class="`diff-line--${line.op}`"><span class="diff-line__mark">{{ MARK[line.op] }}</span>{{ line.text }}</div>
        </div>
        <pre v-else-if="view.kind === 'write'" class="call-detail__box call-detail__code">{{ view.contents }}</pre>
      </template>
      <dl v-if="'fields' in view && view.fields.length" class="call-detail__fields">
        <div v-for="field in view.fields" :key="field.label">
          <dt class="call-detail__label">{{ field.label }}</dt>
          <dd class="call-detail__text" :class="{ 'call-detail__box call-detail__code': field.nested }">{{ field.value }}</dd>
        </div>
      </dl>
      <ol v-if="'pages' in view && view.pages.length" class="call-detail__pages">
        <li v-for="(page, i) in view.pages" :key="i">
          <p v-if="page.title" class="call-detail__text">{{ page.title }}</p>
          <p class="call-detail__label">{{ page.url }}</p>
        </li>
      </ol>
    </template>
    <button v-if="delegate" v-bind="delegatePillAttrs(delegate)">Open transcript</button>
  </div>
</template>
