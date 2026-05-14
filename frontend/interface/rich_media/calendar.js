/**
 * Calendar card module.
 *
 * Implements the rich-media card contract: exports render(payload, synthesis, root).
 *
 * Single event (get_event / update_event):
 *   date block + title + time range + location + attendees
 *
 * Event list (list_events):
 *   compact rows grouped by day
 *
 * Payload shape:
 *   { action_performed, event?: {...}, events?: [...], count? }
 */

const DAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

function parseISO(s) {
  if (!s) return null;
  const d = new Date(s);
  return Number.isNaN(d.getTime()) ? null : d;
}

function formatTime(d) {
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false });
}

function dateKey(d) {
  return `${d.getFullYear()}-${d.getMonth()}-${d.getDate()}`;
}

function buildDateBlock(d) {
  const when = document.createElement('div');
  when.className = 'calendar-card__when';

  const dayEl = document.createElement('div');
  dayEl.className = 'calendar-card__when-day';
  dayEl.textContent = d ? DAYS[d.getDay()] : '—';
  when.appendChild(dayEl);

  const dateEl = document.createElement('div');
  dateEl.className = 'calendar-card__when-date';
  dateEl.textContent = d ? d.getDate() : '—';
  when.appendChild(dateEl);

  const monEl = document.createElement('div');
  monEl.className = 'calendar-card__when-mon';
  monEl.textContent = d ? MONTHS[d.getMonth()] : '';
  when.appendChild(monEl);

  return when;
}

function buildSingleEvent(event) {
  const card = document.createElement('div');
  card.className = 'rich-card calendar-card';

  const dtstart = parseISO(event.dtstart);
  card.appendChild(buildDateBlock(dtstart));

  const info = document.createElement('div');
  info.className = 'calendar-card__info';

  const title = document.createElement('h4');
  title.className = 'calendar-card__title';
  title.textContent = event.title || '';
  info.appendChild(title);

  const meta = document.createElement('div');
  meta.className = 'calendar-card__meta';

  if (event.all_day) {
    const bold = document.createElement('b');
    bold.textContent = 'All day';
    meta.appendChild(bold);
  } else if (dtstart) {
    const dtend = parseISO(event.dtend);
    const bold = document.createElement('b');
    bold.textContent = dtend ? `${formatTime(dtstart)} – ${formatTime(dtend)}` : formatTime(dtstart);
    meta.appendChild(bold);
  }

  if (event.calendar_name) {
    if (meta.firstChild) {
      meta.appendChild(document.createTextNode(' · '));
    }
    meta.appendChild(document.createTextNode(event.calendar_name));
  }

  info.appendChild(meta);

  if (event.location) {
    const loc = document.createElement('div');
    loc.className = 'calendar-card__location';
    loc.textContent = event.location;
    info.appendChild(loc);
  }

  if (Array.isArray(event.attendees) && event.attendees.length > 0) {
    const att = document.createElement('div');
    att.className = 'calendar-card__attendees';
    att.textContent = event.attendees.join(', ');
    info.appendChild(att);
  }

  card.appendChild(info);
  return card;
}

function buildEventList(events) {
  const card = document.createElement('div');
  card.className = 'rich-card calendar-card calendar-card--list';

  const groups = new Map();
  for (const ev of events) {
    const dt = parseISO(ev.dtstart);
    const key = dt ? dateKey(dt) : 'unknown';
    if (!groups.has(key)) groups.set(key, { date: dt, items: [] });
    groups.get(key).items.push(ev);
  }

  for (const [, group] of groups) {
    const section = document.createElement('div');
    section.className = 'calendar-card__day-group';

    if (group.date) {
      const label = document.createElement('div');
      label.className = 'calendar-card__day-label';
      label.textContent = `${DAYS[group.date.getDay()]} ${group.date.getDate()} ${MONTHS[group.date.getMonth()]}`;
      section.appendChild(label);
    }

    for (const ev of group.items) {
      const row = document.createElement('div');
      row.className = 'calendar-card__row';

      const time = document.createElement('span');
      time.className = 'calendar-card__row-time';
      const dtstart = parseISO(ev.dtstart);
      if (ev.all_day) {
        time.textContent = 'All day';
      } else if (dtstart) {
        time.textContent = formatTime(dtstart);
      }
      row.appendChild(time);

      const name = document.createElement('span');
      name.className = 'calendar-card__row-title';
      name.textContent = ev.title || '';
      row.appendChild(name);

      if (ev.location) {
        const loc = document.createElement('span');
        loc.className = 'calendar-card__row-loc';
        loc.textContent = ev.location;
        row.appendChild(loc);
      }

      section.appendChild(row);
    }
    card.appendChild(section);
  }

  return card;
}

export function render(payload, synthesis, root) {
  if (payload.event) {
    root.appendChild(buildSingleEvent(payload.event));
  } else if (Array.isArray(payload.events) && payload.events.length > 0) {
    if (payload.events.length === 1) {
      root.appendChild(buildSingleEvent(payload.events[0]));
    } else {
      root.appendChild(buildEventList(payload.events));
    }
  }
}
