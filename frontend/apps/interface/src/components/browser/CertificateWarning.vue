<!-- The warning a tab shows in place of a page the browser refused for its
     certificate, as any browser does. Going back is the safe choice; only the
     user's click on Proceed lets the browser open the site, and then in every
     tab until it restarts. -->
<script setup lang="ts">
import { computed } from 'vue';
import type { BrowserCertificateError } from '../../api/browser';

const props = defineProps<{ refused: BrowserCertificateError }>();
defineEmits<{ proceed: []; back: [] }>();

/** What the browser's reason means, in words; any reason not listed reads as the general one. */
const REASONS: Record<string, string> = {
  ERR_CERT_AUTHORITY_INVALID: 'has a certificate that no trusted authority issued; it may be self-signed',
  ERR_CERT_COMMON_NAME_INVALID: 'has a certificate made out for a different site',
  ERR_CERT_DATE_INVALID: "has a certificate that has expired or isn't valid yet",
  ERR_CERT_REVOKED: 'has a certificate its issuer has withdrawn',
};

/** Host and port as an address bar writes them: the default port left out, an IPv6 host in brackets. */
const site = computed(() => {
  const { host, port } = props.refused;
  const name = host.includes(':') ? `[${host}]` : host;
  return port === 443 ? name : `${name}:${port}`;
});

const reason = computed(() => REASONS[props.refused.error] ?? "has a certificate the browser can't trust");
</script>

<template>
  <section class="cert-warning" aria-labelledby="cert-warning-title">
    <div class="cert-warning__content">
      <h2 id="cert-warning-title" class="cert-warning__title">This site's certificate isn't trusted</h2>
      <p>
        <b>{{ site }}</b> {{ reason }}, so the browser stopped before opening it. Someone could be
        pretending to be this site to see what you send it.
      </p>
      <p>
        Go on only if you know this site and expected this, such as a device on your own network.
        The browser will then open {{ site }} in every tab, Chalie's included, until it restarts.
      </p>
      <code class="cert-warning__code">{{ refused.error }}</code>
      <div class="cert-warning__actions">
        <button type="button" class="btn btn-primary" @click="$emit('back')">Go back</button>
        <button type="button" class="btn" @click="$emit('proceed')">Proceed to {{ site }}</button>
      </div>
    </div>
  </section>
</template>

<style scoped lang="scss">
.cert-warning {
  height: 100%;
  overflow-y: auto;
  display: flex;
  justify-content: center;
  padding: var(--space-xl) var(--space-lg);
  color: var(--text);
}

.cert-warning__content {
  width: min(560px, 100%);
  overflow-wrap: anywhere;
}

.cert-warning__title {
  margin: 0 0 var(--space-md);
  font: 600 var(--fs-title) var(--font-ui);
}

.cert-warning__code {
  display: block;
  margin: var(--space-md) 0 var(--space-lg);
  color: var(--muted);
  font-size: var(--fs-mono);
}

.cert-warning__actions {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-sm);
}
</style>
