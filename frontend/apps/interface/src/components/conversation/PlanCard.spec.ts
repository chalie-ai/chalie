// @vitest-environment happy-dom
/**
 * PlanCard — the plan's checklist: a head counting the done tasks (the
 * cancelled ones too, for screen readers) over a bar with one segment per
 * task, then each task's title behind a mark that names its status — a
 * dashed circle to do, a spinning arc in progress, a check-circle done, a
 * slashed circle cancelled — with the status also read out for screen readers.
 *
 * A status change swaps the mark and the new one animates in; the card's
 * first render does not animate, so a reload or an opened settled plan shows
 * its marks still. That is pinned through the real <Transition> (the test
 * utils' default transition stub is turned off), reading the enter class the
 * animation hangs on.
 */
import { describe, expect, it } from 'vitest';
import { mount } from '@vue/test-utils';
import PlanCard from './PlanCard.vue';
import type { PlanTask } from '../../api/conversation';

function task(number: number, title: string, status: PlanTask['status']): PlanTask {
  return { number, title, task: `do: ${title}`, status };
}

const TASKS: PlanTask[] = [
  task(1, 'Find your location', 'done'),
  task(2, 'Get today\'s forecast', 'in_progress'),
  task(3, 'Check for rain warnings', 'cancelled'),
  task(4, 'Report the weather', 'todo'),
];

function mountCard(tasks: PlanTask[]) {
  return mount(PlanCard, { props: { tasks }, global: { stubs: { transition: false } } });
}

/** Wait out the frames a transition takes to start, while it still runs. */
const frames = () => new Promise((r) => setTimeout(r, 50));

describe('PlanCard', () => {
  it('counts the done and cancelled tasks in its head, lights a bar segment per task, and lists every task in order, its status on its row', () => {
    const wrapper = mountCard(TASKS);

    expect(wrapper.find('.plan-card__count').text()).toBe('1/4');
    expect(wrapper.find('.plan-card__summary').text()).toBe('1 of 4 done, 1 cancelled');
    expect(wrapper.findAll('.plan-card__bar i').map((s) => s.classes())).toEqual([
      ['plan-card__segment--done'], ['plan-card__segment--in_progress'],
      ['plan-card__segment--cancelled'], ['plan-card__segment--todo'],
    ]);
    const rows = wrapper.findAll('.plan-task');
    expect(rows.map((r) => r.find('.plan-task__title').text())).toEqual(TASKS.map((t) => t.title));
    expect(rows.map((r) => r.classes().find((c) => c.startsWith('plan-task--')))).toEqual([
      'plan-task--done', 'plan-task--in_progress', 'plan-task--cancelled', 'plan-task--todo',
    ]);
    expect(rows.map((r) => r.find('.plan-task__status').text())).toEqual([
      ', done', ', in progress', ', cancelled', ', to do',
    ]);
  });

  it('leaves the cancelled count out of the head when nothing was cancelled', () => {
    const wrapper = mountCard([task(1, 'Find your location', 'done'), task(2, 'Report the weather', 'todo')]);

    expect(wrapper.find('.plan-card__summary').text()).toBe('1 of 2 done');
  });

  it('marks a done task with a check, a running one with a spinner, a cancelled one with a slash, and an open one with a dashed circle', () => {
    const wrapper = mountCard(TASKS);
    const marks = wrapper.findAll('.plan-task__mark');

    expect(marks[0]!.find('.plan-task__glyph').attributes('d')).toBe('m8 12 3 3 5-6');
    expect(marks[1]!.find('.plan-task__spin').exists()).toBe(true);
    expect(marks[2]!.find('.plan-task__glyph').attributes('d')).toBe('M8 16 16 8');
    expect(marks[3]!.findAll('path')).toHaveLength(8);
    expect(marks[3]!.find('circle').exists()).toBe(false);
    // The mark is decoration — the status is read out with the title.
    expect(marks.every((m) => m.attributes('aria-hidden') === 'true')).toBe(true);
  });

  it('draws its marks still on first render, then animates in only the mark whose task changed status', async () => {
    const before = [task(1, 'Find your location', 'in_progress'), task(2, 'Report the weather', 'todo')];
    const wrapper = mountCard(before);
    await frames();

    expect(wrapper.find('.plan-mark-enter-active').exists()).toBe(false);

    const row = wrapper.findAll('.plan-task')[0]!.element;
    await wrapper.setProps({ tasks: [task(1, 'Find your location', 'done'), before[1]!] });
    await frames();

    // The same row, patched in place: its mark swapped to the check, which is
    // animating in; the untouched task's dashed circle is not.
    const rows = wrapper.findAll('.plan-task');
    expect(rows[0]!.element).toBe(row);
    expect(rows[0]!.classes()).toContain('plan-task--done');
    const entering = wrapper.findAll('.plan-mark-enter-active');
    expect(entering).toHaveLength(1);
    expect(entering[0]!.find('.plan-task__glyph').attributes('d')).toBe('m8 12 3 3 5-6');
    expect(rows[1]!.find('.plan-mark-enter-active').exists()).toBe(false);

    // The animation ends on its 320ms enter duration.
    await new Promise((r) => setTimeout(r, 400));
    expect(wrapper.find('.plan-mark-enter-active').exists()).toBe(false);
  });
});
