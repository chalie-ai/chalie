<script setup lang="ts">
import { computed } from 'vue';
import { Mail, Phone } from '@lucide/vue';

export interface ContactPhone {
  value?: string;
  type?: string;
}

export interface ContactEmail {
  value?: string;
  type?: string;
}

export interface Contact {
  fn?: string;
  given_name?: string;
  family_name?: string;
  nickname?: string;
  uid?: string;
  phones?: ContactPhone[];
  emails?: ContactEmail[];
  /** Legacy flat fallbacks (some contacts carry { email, name } instead of arrays). */
  email?: string;
  name?: string;
  org?: string;
  title?: string;
}

export interface ContactsPayload {
  contact?: Contact;
  contacts?: Contact[];
  count?: number;
  action_performed?: string;
}

const props = defineProps<{
  payload: ContactsPayload;
  // synthesis satisfies the shared rich-card prop contract; not rendered here.
  synthesis?: string;
}>();

/**
 * Display initials, by priority: given+family first chars; else first+last
 * word of fn/name; else first char; else '?'. Always uppercased.
 */
function initials(contact: Contact): string {
  const g: string = (contact.given_name ?? '').charAt(0);
  const f: string = (contact.family_name ?? '').charAt(0);
  if (g || f) return (g + f).toUpperCase();

  const fn: string = contact.fn ?? contact.name ?? '';
  const parts: string[] = fn.trim().split(/\s+/);
  if (parts.length >= 2) {
    return (
      (parts[0] as string).charAt(0) + (parts[parts.length - 1] as string).charAt(0)
    ).toUpperCase();
  }
  return fn.charAt(0).toUpperCase() || '?';
}

function primaryValue(arr: Array<{ value?: string }> | undefined): string | null {
  return arr?.[0]?.value ?? null;
}

// payload.contact or a 1-item contacts list → single; ≥2 items → list.
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
  <div v-if="singleContact" class="rich-card ct">
    <div class="ct__avatar">{{ initials(singleContact) }}</div>

    <div class="ct__info">
      <h4 class="ct__name">{{ singleContact.fn ?? singleContact.name ?? '' }}</h4>

      <div
        v-if="[singleContact.title, singleContact.org].filter(Boolean).join(' · ')"
        class="ct__subtitle"
      >
        {{ [singleContact.title, singleContact.org].filter(Boolean).join(' · ') }}
      </div>

      <div
        v-if="
          singleContact.phones?.some((p) => p.value) ||
          singleContact.emails?.some((e) => e.value) ||
          (typeof singleContact.email === 'string' && singleContact.email)
        "
        class="ct__fields"
      >
        <a
          v-for="p in (singleContact.phones ?? []).filter((p) => p.value)"
          :key="`tel-${p.value}`"
          class="ct__field"
          :href="`tel:${p.value}`"
        >
          <Phone :size="11" />
          <span>{{ p.value }}</span>
        </a>

        <a
          v-for="e in (singleContact.emails ?? []).filter((e) => e.value)"
          :key="`mailto-${e.value}`"
          class="ct__field"
          :href="`mailto:${e.value}`"
        >
          <Mail :size="11" />
          <span>{{ e.value }}</span>
        </a>

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

  <div v-else-if="listContacts.length > 0" class="rich-card ct ct--list">
    <div v-for="(c, idx) in listContacts" :key="c.uid ?? c.fn ?? c.name ?? idx" class="ct__row">
      <div class="ct__avatar ct__avatar--sm">{{ initials(c) }}</div>

      <span class="ct__row-name">{{ c.fn ?? c.name ?? '' }}</span>

      <a v-if="primaryValue(c.phones)" class="ct__field" :href="`tel:${primaryValue(c.phones)}`">
        <Phone :size="11" />
        <span>{{ primaryValue(c.phones) }}</span>
      </a>

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
/* .ct-namespaced rules only; base .rich-card chrome lives globally in base_card.css. */

.rich-card.ct {
  max-width: 620px;
}

.ct:not(.ct--list) {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 18px;
  align-items: center;
}

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

.ct__avatar--sm {
  width: 30px;
  height: 30px;
  font-size: 0.7rem;
  letter-spacing: -0.01em;
}

.ct__info {
  min-width: 0;
}

.ct__name {
  font-size: 1.15rem;
  font-weight: 600;
  letter-spacing: -0.01em;
  margin: 0 0 8px;
  color: var(--text-primary);
}

.ct__subtitle {
  font-size: 0.78rem;
  color: var(--text-tertiary);
  margin: -4px 0 8px;
}

.ct__fields {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

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
