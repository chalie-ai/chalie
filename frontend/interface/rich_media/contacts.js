/**
 * Contacts card module.
 *
 * Implements the rich-media card contract: exports render(payload, synthesis, root).
 *
 * Single contact (get):
 *   initials avatar + name + org/title + phone/email/location field pills
 *
 * Contact list (list):
 *   compact rows with small avatar, name, and primary phone/email pills
 *
 * Payload shape:
 *   { action_performed, contact?: {...}, contacts?: [...], count? }
 *
 * Contact fields: { uid, fn, given_name, family_name, nickname,
 *   emails: [{value, type}], phones: [{value, type}], org, title }
 *
 * Legacy contacts may have { email, name } instead of structured fields.
 */

const SVG_NS = 'http://www.w3.org/2000/svg';

function makeSvg(pathData, viewBox = '0 0 24 24') {
  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('width', '11');
  svg.setAttribute('height', '11');
  svg.setAttribute('viewBox', viewBox);
  svg.setAttribute('fill', 'none');
  svg.setAttribute('stroke', 'currentColor');
  svg.setAttribute('stroke-width', '2');
  svg.setAttribute('stroke-linecap', 'round');
  svg.setAttribute('stroke-linejoin', 'round');
  svg.innerHTML = pathData;
  return svg;
}

const ICON_EMAIL = '<rect x="3" y="5" width="18" height="14" rx="2"/><polyline points="3 7 12 13 21 7"/>';
const ICON_PHONE = '<path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"/>';
const ICON_LOCATION = '<path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/>';

function initials(contact) {
  const g = (contact.given_name || '').charAt(0);
  const f = (contact.family_name || '').charAt(0);
  if (g || f) return (g + f).toUpperCase();
  const fn = (contact.fn || contact.name || '');
  const parts = fn.trim().split(/\s+/);
  if (parts.length >= 2) return (parts[0].charAt(0) + parts[parts.length - 1].charAt(0)).toUpperCase();
  return fn.charAt(0).toUpperCase() || '?';
}

function buildAvatar(contact, small = false) {
  const el = document.createElement('div');
  el.className = small ? 'ct__avatar ct__avatar--sm' : 'ct__avatar';
  el.textContent = initials(contact);
  return el;
}

function buildField(href, iconHtml, label) {
  const a = document.createElement('a');
  a.className = 'ct__field';
  if (href) {
    a.href = href;
  } else {
    a.setAttribute('role', 'presentation');
  }
  a.appendChild(makeSvg(iconHtml));
  const text = document.createElement('span');
  text.textContent = label;
  a.appendChild(text);
  return a;
}

function buildSingleContact(contact) {
  const card = document.createElement('div');
  card.className = 'rich-card ct';

  card.appendChild(buildAvatar(contact, false));

  const info = document.createElement('div');
  info.className = 'ct__info';

  const name = document.createElement('h4');
  name.className = 'ct__name';
  name.textContent = contact.fn || contact.name || '';
  info.appendChild(name);

  const subtitle = [contact.title, contact.org].filter(Boolean).join(' · ');
  if (subtitle) {
    const sub = document.createElement('div');
    sub.className = 'ct__subtitle';
    sub.textContent = subtitle;
    info.appendChild(sub);
  }

  const fields = document.createElement('div');
  fields.className = 'ct__fields';

  const phones = Array.isArray(contact.phones) ? contact.phones : [];
  for (const p of phones) {
    if (p.value) fields.appendChild(buildField(`tel:${p.value}`, ICON_PHONE, p.value));
  }

  const emails = Array.isArray(contact.emails) ? contact.emails : [];
  for (const e of emails) {
    if (e.value) fields.appendChild(buildField(`mailto:${e.value}`, ICON_EMAIL, e.value));
  }

  if (typeof contact.email === 'string' && contact.email) {
    fields.appendChild(buildField(`mailto:${contact.email}`, ICON_EMAIL, contact.email));
  }

  if (fields.childElementCount > 0) info.appendChild(fields);

  card.appendChild(info);
  return card;
}

function primaryValue(arr) {
  if (!Array.isArray(arr) || arr.length === 0) return null;
  return arr[0].value || null;
}

function buildContactList(contacts) {
  const card = document.createElement('div');
  card.className = 'rich-card ct ct--list';

  for (const c of contacts) {
    const row = document.createElement('div');
    row.className = 'ct__row';

    row.appendChild(buildAvatar(c, true));

    const name = document.createElement('span');
    name.className = 'ct__row-name';
    name.textContent = c.fn || c.name || '';
    row.appendChild(name);

    const phone = primaryValue(c.phones);
    if (phone) {
      row.appendChild(buildField(`tel:${phone}`, ICON_PHONE, phone));
    }

    const email = primaryValue(c.emails) || c.email;
    if (email) {
      row.appendChild(buildField(`mailto:${email}`, ICON_EMAIL, email));
    }

    card.appendChild(row);
  }

  return card;
}

export function render(payload, synthesis, root) {
  if (payload.contact) {
    root.appendChild(buildSingleContact(payload.contact));
  } else if (Array.isArray(payload.contacts) && payload.contacts.length > 0) {
    if (payload.contacts.length === 1) {
      root.appendChild(buildSingleContact(payload.contacts[0]));
    } else {
      root.appendChild(buildContactList(payload.contacts));
    }
  }
}
