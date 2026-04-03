#!/usr/bin/env node

const { execSync } = require('child_process');
const https = require('https');
const fs = require('fs');

const VERSION = process.env.RELEASE_VERSION; // e.g. "v0.3.1"

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
    return null;
  }
}

function getCommitsSince(ref) {
  try {
    const range = ref ? `${ref}..HEAD` : 'HEAD~30..HEAD';
    const output = execSync(
      `git log ${range} --pretty=format:"%s%n%b" --no-merges`,
      { encoding: 'utf8' }
    );
    return output.trim();
  } catch (e) {
    return '';
  }
}

function callGeminiOnce(systemPrompt, userPrompt, model) {
  return new Promise((resolve, reject) => {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) return reject(new Error('GEMINI_API_KEY not set'));

    const payload = JSON.stringify({
      contents: [{ parts: [{ text: userPrompt }] }],
      system_instruction: { parts: [{ text: systemPrompt }] }
    });
    const buf = Buffer.from(payload, 'utf8');

    const req = https.request({
      hostname: 'generativelanguage.googleapis.com',
      port: 443,
      path: `/v1beta/models/${model}:generateContent?key=${apiKey}`,
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': buf.length }
    }, (res) => {
      let data = '';
      res.on('data', c => data += c);
      res.on('end', () => {
        if (res.statusCode >= 200 && res.statusCode < 300) {
          try {
            const r = JSON.parse(data);
            const text = r.candidates?.[0]?.content?.parts?.[0]?.text?.trim();
            if (!text) return reject(new Error('Empty Gemini response'));
            resolve(text.replace(/^["']|["']$/g, '').trim());
          } catch (e) {
            reject(new Error(`Parse error: ${e.message}`));
          }
        } else {
          const err = new Error(`Gemini API ${res.statusCode}: ${data}`);
          err.statusCode = res.statusCode;
          reject(err);
        }
      });
    });
    req.on('error', reject);
    req.write(buf);
    req.end();
  });
}

async function callGemini(systemPrompt, userPrompt) {
  const primaryModel = process.env.GEMINI_MODEL || 'gemini-2.0-flash';
  const fallbackModel = 'gemini-2.5-pro';
  const delays = [3000, 10000, 30000];

  for (const model of [primaryModel, fallbackModel]) {
    let lastErr;
    for (let attempt = 0; attempt <= delays.length; attempt++) {
      try {
        return await callGeminiOnce(systemPrompt, userPrompt, model);
      } catch (err) {
        lastErr = err;
        const retryable = err.statusCode >= 500 || err.statusCode === 429;
        if (retryable && attempt < delays.length) {
          console.log(`Gemini ${err.statusCode} (${model}), retrying in ${delays[attempt] / 1000}s... (attempt ${attempt + 1}/${delays.length})`);
          await new Promise(r => setTimeout(r, delays[attempt]));
        } else {
          break;
        }
      }
    }
    if (model === primaryModel) {
      console.log(`Primary model ${primaryModel} failed, trying fallback ${fallbackModel}...`);
    } else {
      throw lastErr;
    }
  }
}

async function postToZapier(webhookUrl, tweet) {
  return new Promise((resolve, reject) => {
    const body = JSON.stringify({ version: VERSION, text: tweet });
    const buf = Buffer.from(body, 'utf8');
    const url = new URL(webhookUrl);

    const req = https.request({
      hostname: url.hostname,
      port: 443,
      path: url.pathname + url.search,
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': buf.length }
    }, (res) => {
      let data = '';
      res.on('data', c => data += c);
      res.on('end', () => {
        if (res.statusCode >= 200 && res.statusCode < 300) {
          resolve();
        } else {
          reject(new Error(`Zapier webhook ${res.statusCode}: ${data}`));
        }
      });
    });
    req.on('error', reject);
    req.write(buf);
    req.end();
  });
}

function writeSummary(tweet) {
  const summaryFile = process.env.GITHUB_STEP_SUMMARY;
  if (!summaryFile) return;
  fs.appendFileSync(summaryFile, `## 🚀 ${VERSION} release post\n\n> ${tweet}\n`);
}

async function main() {
  const prevTag = getPreviousTag();
  const commits = getCommitsSince(prevTag);

  console.log(`Generating release post for ${VERSION} (since ${prevTag || 'beginning'})`);

  const systemPrompt = `You are the voice of Chalie — a personal AI assistant with a dry, confident personality.
Write a single social media post announcing a new release. Rules:
- First person ("I", "my", not "we")
- Max 240 characters
- Include the version number (${VERSION})
- 1–3 relevant emojis, placed naturally (not as a trailing dump)
- Punchy and sassy but still descriptive of what actually changed
- No hashtags, no links, no quotes around the post
- Vary the structure — don't always lead with the version number

Good examples:
${VERSION} is out — fixed a few things that were quietly annoying me 🔧
I can now read your emails in ${VERSION}. You're welcome 📬
Got eyes. ${VERSION} ships OCR support 👁️
${VERSION} — contacts are live if you've plugged me into Gmail 📇`;

  const userPrompt = commits
    ? `Commits in this release:\n${commits.slice(0, 3000)}`
    : `Version ${VERSION} was just released.`;

  const post = await callGemini(systemPrompt, userPrompt);
  console.log(`Post (${post.length} chars): ${post}`);

  if (post.length > 280) {
    throw new Error(`Post too long: ${post.length} chars`);
  }

  writeSummary(post);

  const webhookUrl = process.env.ZAPIER_WEBHOOK_URL;
  if (!webhookUrl) {
    console.log('ZAPIER_WEBHOOK_URL not set — skipping webhook. Post written to summary.');
    return;
  }

  await postToZapier(webhookUrl, post);
  console.log('Zapier webhook fired successfully.');
}

main().catch(err => {
  console.error(err.message);
  process.exit(1);
});
