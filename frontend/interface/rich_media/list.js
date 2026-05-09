/**
 * List card module.
 *
 * Implements the rich-media card contract: exports render(payload, synthesis, root).
 * Renders a checklist with progress bar and click-to-toggle items.
 *
 * Payload shape (from ListAbility):
 *   { status: "success", list: { name, items: [{ content, checked }] } }
 */

const CHECK_SVG = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>';

function updateProgress(card) {
  const items = card.querySelectorAll('.list-card__item');
  const done = card.querySelectorAll('.list-card__item--done');
  const total = items.length;
  const doneCount = done.length;

  const progressText = card.querySelector('.list-card__progress');
  if (progressText) {
    progressText.innerHTML = `<b>${doneCount}</b> / ${total} done`;
  }

  const barFill = card.querySelector('.list-card__bar-fill');
  if (barFill) {
    barFill.style.width = total > 0 ? `${(doneCount / total) * 100}%` : '0%';
  }
}

/**
 * Build and mount the list card DOM into root.
 */
export function render(payload, synthesis, root) {
  const listData = payload.list || payload;
  const items = listData.items || [];
  const name = listData.name || '';
  const id = listData.id || '';
  const checkedCount = items.filter(i => i.checked).length;

  const card = document.createElement('div');
  card.className = 'rich-card';

  // Header
  const head = document.createElement('div');
  head.className = 'list-card__head';

  const title = document.createElement('h4');
  title.className = 'list-card__title';
  title.textContent = name;
  head.appendChild(title);

  const progress = document.createElement('div');
  progress.className = 'list-card__progress';
  progress.innerHTML = `<b>${checkedCount}</b> / ${items.length} done`;
  head.appendChild(progress);

  card.appendChild(head);

  // Progress bar
  const bar = document.createElement('div');
  bar.className = 'list-card__bar';
  const barFill = document.createElement('div');
  barFill.className = 'list-card__bar-fill';
  barFill.style.width = items.length > 0 ? `${(checkedCount / items.length) * 100}%` : '0%';
  bar.appendChild(barFill);
  card.appendChild(bar);

  // Items
  const itemsContainer = document.createElement('div');
  itemsContainer.className = 'list-card__items';

  for (const item of items) {
    const row = document.createElement('div');
    row.className = 'list-card__item' + (item.checked ? ' list-card__item--done' : '');

    const check = document.createElement('span');
    check.className = 'list-card__check';
    check.innerHTML = CHECK_SVG;
    row.appendChild(check);

    const text = document.createElement('div');
    text.className = 'list-card__text';
    text.textContent = item.content || '';
    row.appendChild(text);

    row.addEventListener('click', () => {
      if (row.dataset.busy === '1') return;
      const wasDone = row.classList.contains('list-card__item--done');
      const nextChecked = !wasDone;

      // Optimistic flip — revert on backend error.
      row.classList.toggle('list-card__item--done');
      updateProgress(card);
      row.dataset.busy = '1';

      const revert = () => {
        row.classList.toggle('list-card__item--done');
        updateProgress(card);
        delete row.dataset.busy;
      };

      document.dispatchEvent(new CustomEvent('chalie:silent-action', {
        detail: {
          payload: {
            skill: 'list',
            action: 'check',
            id: id,
            items: [{ content: item.content, checked: nextChecked }],
          },
          onMessage: () => { delete row.dataset.busy; },
          onError: revert,
          onDone: () => { delete row.dataset.busy; },
        },
      }));
    });

    itemsContainer.appendChild(row);
  }

  card.appendChild(itemsContainer);
  root.appendChild(card);
}
