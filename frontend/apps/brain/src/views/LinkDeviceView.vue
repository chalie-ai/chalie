<script setup lang="ts">
// Generates a pairing QR for the Chalie mobile app: mints a one-shot bearer
// token, pairs it with this instance's own origin, and encodes the locked
// PairingPayload as a scannable QR shown in a modal. The raw token is shown
// ONCE (server never stores it) — re-minting issues a new one.
import { nextTick, ref } from 'vue';
import qrcode from 'qrcode-generator';
import type { PairingPayload } from '@chalie/shared';
import { validatePairingPayload } from '@chalie/shared';
import { wrappers } from '../api/wrappers';
import { system } from '../api/system';
import { readBrainOrigin } from '../api/origin';
import { useToast } from '../composables/useToast';
import { apiErrorMessage } from '../api/http';
import BrainModal from '../ui/BrainModal.vue';
import { QrCode, Smartphone, X } from '@lucide/vue';

const { show: showToast } = useToast();

const qrCanvas = ref<HTMLCanvasElement | null>(null);
// The minted payload, JSON-stringified — bound to data-pairing for the
// feature test to decode and assert the contract end-to-end.
const pairingJson = ref('');
const minting = ref(false);
const showQr = ref(false);

// Quiet-zone modules on each side of the symbol.
const QR_MARGIN = 2;

// Renders `text` as a QR symbol filling the canvas: the module grid is scaled to
// the canvas width including the quiet zone, and each module is snapped to whole
// pixels so nothing is anti-aliased into an unscannable smear.
function drawQr(canvas: HTMLCanvasElement, text: string): void {
  qrcode.stringToBytes = qrcode.stringToBytesFuncs['UTF-8'];
  const qr = qrcode(0, 'M');
  qr.addData(text, 'Byte');
  qr.make();

  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('QR canvas is not available.');

  const size = canvas.width;
  const count = qr.getModuleCount();
  const scale = size / (count + QR_MARGIN * 2);

  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, size, size);
  ctx.fillStyle = '#000000';
  for (let row = 0; row < count; row += 1) {
    const y = Math.ceil((row + QR_MARGIN) * scale);
    const height = Math.ceil((row + QR_MARGIN + 1) * scale) - y;
    for (let col = 0; col < count; col += 1) {
      if (!qr.isDark(row, col)) continue;
      const x = Math.ceil((col + QR_MARGIN) * scale);
      ctx.fillRect(x, y, Math.ceil((col + QR_MARGIN + 1) * scale) - x, height);
    }
  }
}

async function generate(): Promise<void> {
  if (minting.value) return;
  minting.value = true;
  showToast('Minting pairing token…', 'info');
  try {
    const host = readBrainOrigin();
    const [{ token }, { username }] = await Promise.all([
      wrappers.create({ name: `Mobile — ${new Date().toISOString().slice(0, 10)}` }),
      system.username(),
    ]);
    const payload: PairingPayload = {
      v: 1,
      host,
      token,
      username,
    };
    validatePairingPayload(payload); // contract gate — throws on a bad payload.
    const json = JSON.stringify(payload);
    showQr.value = true;
    await nextTick(); // the canvas only mounts once the modal is open.
    const canvas = qrCanvas.value;
    if (!canvas) throw new Error('QR canvas not ready.');
    drawQr(canvas, json);
    pairingJson.value = json;
    showToast('Scan this QR with the Chalie app', 'success', { duration: 8000 });
  } catch (e) {
    pairingJson.value = '';
    showQr.value = false;
    showToast(apiErrorMessage(e, 'Could not generate pairing code'), 'error');
  } finally {
    minting.value = false;
  }
}
</script>

<template>
  <div class="panel-header">
    <h2><Smartphone :size="20" /> Link device</h2>
  </div>

  <div class="brain-overview">
    <div class="export-card">
      <div class="export-card-icon"><QrCode :size="24" /></div>
      <div class="export-card-label">Pair the Chalie mobile app</div>
      <p class="form-hint">
        Generates a one-time QR code that links your phone to this instance. It encodes this
        instance's address and a fresh access token. The token is shown <strong>once</strong> —
        re-generate to issue a new one (and revoke the old device).
      </p>
      <button
        class="btn btn-primary"
        :disabled="minting"
        data-action="generate-pairing"
        @click="generate"
      >
        <QrCode :size="14" /> {{ minting ? 'Generating…' : 'Generate pairing code' }}
      </button>
    </div>
  </div>

  <BrainModal v-model="showQr" size="sm">
    <div class="modal-header">
      <h3>Scan to pair</h3>
      <button class="btn-close" data-action="close-pairing" @click="showQr = false">
        <X :size="16" />
      </button>
    </div>
    <div class="pairing-modal-body">
      <p class="form-hint">
        Open the Chalie app on your phone and scan this code. It expires once used — re-generate to
        link another device.
      </p>
      <canvas
        ref="qrCanvas"
        class="pairing-qr"
        data-testid="pairing-qr"
        :data-pairing="pairingJson"
        width="256"
        height="256"
      />
    </div>
  </BrainModal>
</template>

<style scoped>
.pairing-modal-body {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--space-md);
}

.pairing-qr {
  border-radius: var(--radius-md);
  background: var(--bg-surface-2);
  padding: var(--space-md);
  /* Border-box sizing: pins the padded element to the canvas' own 256px so the
     padding insets the symbol rather than growing the element around it. */
  width: 256px;
  height: 256px;
}
</style>
