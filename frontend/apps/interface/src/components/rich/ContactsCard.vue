<!--
  ContactsCard — faithful 1:1 Vue 3 port of contacts.js + contacts.css.

  Single contact (payload.contact or payload.contacts with one item):
    initials avatar + name + org/title subtitle + phone/email field pills

  Contact list (payload.contacts with two or more items):
    compact rows: small avatar + name + primary phone/email pills

  The .rich-card base chrome (background, border, padding, animation) is
  declared globally in base_card.css — not redeclared here.

  Class namespace: .ct (exact match to contacts.js output — not contacts-card /
  contact-item / contact-field-pill, which the source does not emit).
-->
<script setup lang="ts">
import { computed } from 'vue';
import { Mail, Phone } from '@lucide/vue';

// ── Payload contract ───────────────────────────────────────────────────────

/** A single phone entry as used by the legacy contacts vCard model. */
export interface ContactPhone {
  value?: string;
  type?: string;
}

/** A single email entry as used by the legacy contacts vCard model. */
export interface ContactEmail {
  value?: string;
  type?: string;
}

/** All fields contacts.js reads from a contact object. */
export interface Contact {
  /** vCard FN (formatted name). */
  fn?: string;
  /** Given / first name. */
  given_name?: string;
  /** Family / last name. */
  family_name?: string;
  /** Nickname — present in the payload shape comment but not read by the renderer. */
  nickname?: string;
  /** Unique identifier — present in payload but not rendered. */
  uid?: string;
  /** Structured phone array (preferred). */
  phones?: ContactPhone[];
  /** Structured email array (preferred). */
  emails?: ContactEmail[];
  /** Legacy flat email string (legacy contacts may carry { email, name }). */
  email?: string;
  /** Legacy flat name string (legacy contacts may carry { email, name }). */
  name?: string;
  /** Organisation name. */
  org?: string;
  /** Job title. */
  title?: string;
}

/** Top-level payload handed to this card. */
export interface ContactsPayload {
  /** Single contact (get action). */
  contact?: Contact;
  /** Contact list (list action). */
  contacts?: Contact[];
  /** Result count — present in payload shape but not rendered. */
  count?: number;
  /** Action identifier — present in payload shape but not rendered. */
  action_performed?: string;
}

// ── Props ──────────────────────────────────────────────────────────────────

const props = defineProps<{
  payload: ContactsPayload;
  /**
   * synthesis is accepted to satisfy the shared rich-card prop contract but
   * the legacy contacts.js renderer does not output synthesis into the DOM.
   * It is intentionally unused here.
   */
  synthesis?: string;
}>();

// ── Initials logic (exact port of initials() from contacts.js lines 41-49) ─

/**
 * Derives display initials for a contact.
 *
 * Priority (faithful port of initials() in contacts.js):
 *  1. given_name[0] + family_name[0]  — if either part is present
 *  2. fn || name → split on whitespace; if ≥ 2 words: first[0] + last[0]
 *  3. fn || name → first character
 *  4. '?' fallback
 *
 * All results are uppercased.
 */
function initials(contact: Contact): string {
  const g: string = (contact.given_name ?? '').charAt(0);
  const f: string = (contact.family_name ?? '').charAt(0);
  if (g || f) return (g + f).toUpperCase();

  const fn: string = contact.fn ?? contact.name ?? '';
  const parts: string[] = fn.trim().split(/\s+/);
  if (parts.length >= 2) {
    return (
      (parts[0] as string).charAt(0) +
      (parts[parts.length - 1] as string).charAt(0)
    ).toUpperCase();
  }
  return fn.charAt(0).toUpperCase() || '?';
}

// ── Primary value helper (exact port of primaryValue() from contacts.js) ───

/**
 * Returns the first non-empty .value from an array, or null.
 * Exact port of primaryValue() in contacts.js lines 118-121.
 */
function primaryValue(arr: Array<{ value?: string }> | undefined): string | null {
  return arr?.[0]?.value ?? null;
}

// ── Render mode ────────────────────────────────────────────────────────────

/**
 * Resolves which contact(s) to render and in which layout.
 *
 * Mirrors the branching in render() from contacts.js lines 154-163:
 *   payload.contact                        → single
 *   payload.contacts with exactly 1 item  → single (that item)
 *   payload.contacts with ≥ 2 items       → list
 */
const singleContact = computed<Contact | null>(() => {
  if (props.payload.contact) return props.payload.contact;
  const cs = props.payload.contacts;
  return Array.isArray(cs) && cs.length === 1 ? (cs[0] as Contact) : null;
});

const listContacts = computed<Contact[]>(() => {
  const cs = props.payload.contacts;
  return !props.payload.contact && Array.isArray(cs) && cs.length > 1 ? cs : [];
});
</script>

<template>
  <!-- ── Single contact layout ─────────────────────────────────────────────
       Root class: "rich-card ct" — mirrors buildSingleContact() line 76.
       Grid layout (avatar | info) is applied by .ct:not(.ct--list) in SCSS.
  -->
  <div v-if="singleContact" class="rich-card ct">
    <!-- Avatar — .ct__avatar (not small) -->
    <div class="ct__avatar">{{ initials(singleContact) }}</div>

    <!-- Info block -->
    <div class="ct__info">
      <!-- Name — h4.ct__name, textContent = fn || name (contacts.js line 84) -->
      <h4 class="ct__name">{{ singleContact.fn ?? singleContact.name ?? '' }}</h4>

      <!-- Subtitle: title · org — only rendered when non-empty (lines 87-93) -->
      <div
        v-if="[singleContact.title, singleContact.org].filter(Boolean).join(' · ')"
        class="ct__subtitle"
      >
        {{ [singleContact.title, singleContact.org].filter(Boolean).join(' · ') }}
      </div>

      <!-- Fields container — only rendered when it has children (line 112) -->
      <div
        v-if="
          singleContact.phones?.some((p) => p.value) ||
          singleContact.emails?.some((e) => e.value) ||
          (typeof singleContact.email === 'string' && singleContact.email)
        "
        class="ct__fields"
      >
        <!-- Phone pills (contacts.js lines 98-101) -->
        <a
          v-for="p in (singleContact.phones ?? []).filter((p) => p.value)"
          :key="`tel-${p.value}`"
          class="ct__field"
          :href="`tel:${p.value}`"
        >
          <Phone :size="11" />
          <span>{{ p.value }}</span>
        </a>

        <!-- Email pills from structured emails array (contacts.js lines 103-106) -->
        <a
          v-for="e in (singleContact.emails ?? []).filter((e) => e.value)"
          :key="`mailto-${e.value}`"
          class="ct__field"
          :href="`mailto:${e.value}`"
        >
          <Mail :size="11" />
          <span>{{ e.value }}</span>
        </a>

        <!-- Legacy flat email string (contacts.js lines 108-110) -->
        <a
          v-if="typeof singleContact.email === 'string' && singleContact.email"
          class="ct__field"
          :href="`mailto:${singleContact.email}`"
        >
          <Mail :size="11" />
          <span>{{ singleContact.email }}</span>
        </a>
      </div>
    </div>
  </div>

  <!-- ── List layout ───────────────────────────────────────────────────────
       Root class: "rich-card ct ct--list" — mirrors buildContactList() line 125.
  -->
  <div v-else-if="listContacts.length > 0" class="rich-card ct ct--list">
    <div
      v-for="(c, idx) in listContacts"
      :key="c.uid ?? c.fn ?? c.name ?? idx"
      class="ct__row"
    >
      <!-- Small avatar — .ct__avatar.ct__avatar--sm (contacts.js line 131) -->
      <div class="ct__avatar ct__avatar--sm">{{ initials(c) }}</div>

      <!-- Row name — span.ct__row-name (contacts.js line 133) -->
      <span class="ct__row-name">{{ c.fn ?? c.name ?? '' }}</span>

      <!-- Primary phone pill (contacts.js lines 138-141) -->
      <a
        v-if="primaryValue(c.phones)"
        class="ct__field"
        :href="`tel:${primaryValue(c.phones)}`"
      >
        <Phone :size="11" />
        <span>{{ primaryValue(c.phones) }}</span>
      </a>

      <!-- Primary email pill: structured array first, then flat string
           (contacts.js lines 143-145: primaryValue(c.emails) || c.email) -->
      <a
        v-if="primaryValue(c.emails) || c.email"
        class="ct__field"
        :href="`mailto:${primaryValue(c.emails) ?? c.email}`"
      >
        <Mail :size="11" />
        <span>{{ primaryValue(c.emails) ?? c.email }}</span>
      </a>
    </div>
  </div>
</template>

<style scoped lang="scss">
/*
 * Contacts-card-specific rules only (.ct namespace).
 * .rich-card base chrome (bg, border, padding, animation) lives in base_card.css (global).
 * All color values use CSS custom properties — works in both light and dark themes (Rule 7).
 * Exact port of contacts.css; no rules omitted or added beyond scoping adjustments.
 */

.rich-card.ct {
  max-width: 620px;
}

/* ── Single contact layout ─────────────────────────────────────────── */

.ct:not(.ct--list) {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 18px;
  align-items: center;
}

/* Avatar */

.ct__avatar {
  width: 56px;
  height: 56px;
  border-radius: 50%;
  background: color-mix(in oklab, var(--violet) 18%, var(--bg-2));
  border: 1px solid color-mix(in oklab, var(--violet) 35%, transparent);
  color: var(--violet);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.4rem;
  font-weight: 600;
  letter-spacing: -0.02em;
  flex-shrink: 0;
  overflow: hidden;
}

.ct__avatar img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  border-radius: 50%;
}

/* Small avatar (list rows) */

.ct__avatar--sm {
  width: 30px;
  height: 30px;
  font-size: 0.7rem;
  letter-spacing: -0.01em;
}

/* Info block */

.ct__info {
  min-width: 0;
}

/* Name */

.ct__name {
  font-size: 1.15rem;
  font-weight: 600;
  letter-spacing: -0.01em;
  margin: 0 0 8px;
  color: var(--text-primary);
}

/* Org / title subtitle */

.ct__subtitle {
  font-size: 0.78rem;
  color: var(--text-tertiary);
  margin: -4px 0 8px;
}

/* Fields container */

.ct__fields {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

/* Individual field pill */

.ct__field {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: var(--bg-input);
  font-family: var(--font-mono, 'JetBrains Mono', ui-monospace, monospace);
  font-size: 0.74rem;
  color: var(--text-secondary);
  text-decoration: none;
  letter-spacing: 0.02em;
  transition: all 160ms ease;

  &:hover {
    border-color: var(--border-strong);
    color: var(--text-primary);
  }

  svg {
    color: var(--text-tertiary);
    flex-shrink: 0;
  }
}

/* ── List layout ───────────────────────────────────────────────────── */

.ct--list {
  display: flex;
  flex-direction: column;
  gap: 0;
}

.ct__row {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 8px 0;
  border-bottom: 1px solid var(--border);

  &:last-child {
    border-bottom: none;
  }
}

.ct__row-name {
  flex: 1;
  min-width: 80px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 0.88rem;
  font-weight: 500;
  color: var(--text-primary);
}
</style>
