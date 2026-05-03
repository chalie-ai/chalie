/**
 * Scheduler card module.
 *
 * Implements the rich-media card contract: exports render(payload, synthesis, root).
 * Renders a date block + event title + time meta + Confirm button.
 *
 * Payload shape (from ScheduleAbility):
 *   { status, action_performed, record: { id, message, due_at, item_type, recurrence } }
 */

const DAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

function parseDueAt(dueAtStr) {
  if (!dueAtStr) return null;
  const d = new Date(dueAtStr);
  if (isNaN(d.getTime())) return null;
  return d;
}

function formatTime(d) {
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false });
}

/**
 * Build and mount the scheduler card DOM into root.
 */
export function render(payload, synthesis, root) {
  const record = payload.record || payload;
  const dueAt = parseDueAt(record.due_at);

  const card = document.createElement('div');
  card.className = 'rich-card scheduler-card';

  // Date block
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

  // Title + meta
  const info = document.createElement('div');

  const title = document.createElement('h4');
  title.className = 'scheduler-card__title';
  title.textContent = record.message || synthesis || '';
  info.appendChild(title);

  const meta = document.createElement('div');
  meta.className = 'scheduler-card__meta';
  const metaParts = [];
  if (dueAt) {
    const b = document.createElement('b');
    b.textContent = formatTime(dueAt);
    meta.appendChild(b);
    metaParts.push(b);
  }
  const textParts = [];
  if (record.recurrence) textParts.push(record.recurrence);
  if (record.item_type === 'prompt') textParts.push('prompt');
  if (textParts.length > 0) {
    const sep = dueAt ? ' · ' : '';
    meta.appendChild(document.createTextNode(sep + textParts.join(' · ')));
  }
  info.appendChild(meta);

  card.appendChild(info);

  // Confirm button (only for create actions)
  if (payload.action_performed === 'create' || !payload.action_performed) {
    const btn = document.createElement('button');
    btn.className = 'scheduler-card__btn';
    btn.textContent = 'Confirm';
    btn.addEventListener('click', () => {
      btn.textContent = '✓';
      btn.disabled = true;
      btn.style.opacity = '0.6';
    });
    card.appendChild(btn);
  }

  root.appendChild(card);
}
