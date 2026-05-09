/**
 * Scheduler card module.
 *
 * Implements the rich-media card contract: exports render(payload, synthesis, root).
 * Renders a date block + event title + time meta, plus an optional list of
 * other pending items on the same local day.
 *
 * Payload shape (from ScheduleAbility):
 *   { status, action_performed, record: { id, message, due_at, item_type, recurrence },
 *     same_day_items?: [{ id, message, due_at, recurrence }] }
 */

const DAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

function parseDueAt(dueAtStr) {
  if (!dueAtStr) return null;
  const d = new Date(dueAtStr);
  if (Number.isNaN(d.getTime())) return null;
  return d;
}

function formatTime(d) {
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false });
}

function buildMeta(dueAt, record) {
  const meta = document.createElement('div');
  meta.className = 'scheduler-card__meta';
  if (dueAt) {
    const b = document.createElement('b');
    b.textContent = formatTime(dueAt);
    meta.appendChild(b);
  }
  const textParts = [];
  if (record.recurrence) textParts.push(record.recurrence);
  if (record.item_type === 'prompt') textParts.push('prompt');
  if (textParts.length > 0) {
    const sep = dueAt ? ' · ' : '';
    meta.appendChild(document.createTextNode(sep + textParts.join(' · ')));
  }
  return meta;
}

function buildSameDayList(items) {
  const wrap = document.createElement('div');
  wrap.className = 'scheduler-card__same-day';

  const label = document.createElement('div');
  label.className = 'scheduler-card__same-day-label';
  label.textContent = `Also on this day · ${items.length}`;
  wrap.appendChild(label);

  for (const item of items) {
    const row = document.createElement('div');
    row.className = 'scheduler-card__same-day-item';

    const text = document.createElement('span');
    text.className = 'scheduler-card__same-day-text';
    text.textContent = item.message || '';
    row.appendChild(text);

    const dueAt = parseDueAt(item.due_at);
    if (dueAt) {
      const time = document.createElement('span');
      time.className = 'scheduler-card__same-day-time';
      time.textContent = formatTime(dueAt);
      row.appendChild(time);
    }

    wrap.appendChild(row);
  }
  return wrap;
}

/**
 * Build and mount the scheduler card DOM into root.
 */
export function render(payload, synthesis, root) {
  const record = payload.record || payload;
  const dueAt = parseDueAt(record.due_at);

  const card = document.createElement('div');
  card.className = 'rich-card scheduler-card';

  const when = document.createElement('div');
  when.className = 'scheduler-card__when';

  const dayEl = document.createElement('div');
  dayEl.className = 'scheduler-card__when-day';
  dayEl.textContent = dueAt ? DAYS[dueAt.getDay()] : '—';
  when.appendChild(dayEl);

  const dateEl = document.createElement('div');
  dateEl.className = 'scheduler-card__when-date';
  dateEl.textContent = dueAt ? dueAt.getDate() : '—';
  when.appendChild(dateEl);

  const monEl = document.createElement('div');
  monEl.className = 'scheduler-card__when-mon';
  monEl.textContent = dueAt ? MONTHS[dueAt.getMonth()] : '';
  when.appendChild(monEl);

  card.appendChild(when);

  const info = document.createElement('div');
  info.className = 'scheduler-card__info';

  const title = document.createElement('h4');
  title.className = 'scheduler-card__title';
  title.textContent = record.message || synthesis || '';
  info.appendChild(title);

  info.appendChild(buildMeta(dueAt, record));
  card.appendChild(info);

  const sameDay = Array.isArray(payload.same_day_items) ? payload.same_day_items : [];
  if (sameDay.length > 0) {
    card.appendChild(buildSameDayList(sameDay));
  }

  root.appendChild(card);
}
