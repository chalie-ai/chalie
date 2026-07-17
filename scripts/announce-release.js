import {execSync} from 'node:child_process';
import https from 'node:https';
import http from 'node:http';

const VERSION = process.env.RELEASE_VERSION;

if (!VERSION) {
  console.error('RELEASE_VERSION env var is required');
  process.exit(1);
}

function getPreviousTag() {
  try {
    const tags = execSync('git tag --sort=-v:refname', { encoding: 'utf8' })
      .trim().split('\n').filter(Boolean);
    return tags[1] || null;
  } catch (e) {
    console.warn('[announce-release] git tag failed:', e.message);
    return null;
  }
}

function getCommitsSince(ref) {
  try {
    const range = ref ? `${ref}..HEAD` : 'HEAD~30..HEAD';
    const output = execSync(
      `git log ${range} --pretty=format:"%h %s" --no-merges`,
      { encoding: 'utf8' }
    );
    return output.trim().split('\n').filter(Boolean);
  } catch (e) {
    console.warn('[announce-release] git log failed:', e.message);
    return [];
  }
}

async function fireWebhook(payload) {
  return new Promise((resolve, reject) => {
    const webhookUrl = process.env.N8N_ANNOUNCE_WEBHOOK_URL;
    if (!webhookUrl) return reject(new Error('N8N_ANNOUNCE_WEBHOOK_URL not set'));

    const body = JSON.stringify(payload);
    const buf = Buffer.from(body, 'utf8');
    const url = new URL(webhookUrl);
    const lib = url.protocol === 'https:' ? https : http;

    const req = lib.request({
      hostname: url.hostname,
      port: url.port || (url.protocol === 'https:' ? 443 : 80),
      path: url.pathname + url.search,
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': buf.length }
    }, (res) => {
      res.resume();
      res.on('end', () => resolve(res.statusCode));
    });

    // Fire-and-forget: n8n responds immediately, Ollama runs async
    req.setTimeout(10000, () => {
      req.destroy();
      console.log('Webhook delivered (n8n acknowledged, processing async)');
      resolve(202);
    });

    req.on('error', (e) => {
      console.error(`Webhook error (non-fatal): ${e.message}`);
      resolve(0);
    });

    req.write(buf);
    req.end();
  });
}

// Non-fatal — a broken announce must never block the release pipeline
try {
  const prevTag = getPreviousTag();
  const commits = getCommitsSince(prevTag);

  console.log(`Announcing ${VERSION} (${commits.length} commits since ${prevTag || 'beginning'})`);

  const status = await fireWebhook({ version: VERSION, commits });
  console.log(`Webhook status: ${status}`);
} catch (err) {
  console.error(`announce-release failed (non-fatal): ${err.message}`);
  process.exit(0);
}
