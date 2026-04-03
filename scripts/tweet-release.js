#!/usr/bin/env node

const { execSync } = require('child_process');
const https = require('https');
const crypto = require('crypto');

const VERSION = process.env.RELEASE_VERSION; // e.g. "v0.3.1"

if (!VERSION) {
  console.error('RELEASE_VERSION env var is required');
  process.exit(1);
}

function getPreviousTag() {
  try {
    const tags = execSync('git tag --sort=-v:refname', { encoding: 'utf8' })
      .trim().split('\n').filter(Boolean);
    // tags[0] is the current tag, tags[1] is the previous
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

async function callGemini(systemPrompt, userPrompt) {
  return new Promise((resolve, reject) => {
    const apiKey = process.env.GEMINI_API_KEY;
    const model = process.env.GEMINI_MODEL || 'gemini-2.0-flash';

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
          reject(new Error(`Gemini API ${res.statusCode}: ${data}`));
        }
      });
    });
    req.on('error', reject);
    req.write(buf);
    req.end();
  });
}

// Twitter API v2 OAuth 1.0a — no external dependencies
function percentEncode(str) {
  return encodeURIComponent(String(str))
    .replace(/!/g, '%21').replace(/'/g, '%27')
    .replace(/\(/g, '%28').replace(/\)/g, '%29')
    .replace(/\*/g, '%2A');
}

function buildOAuthHeader(method, url, bodyParams) {
  const oauthParams = {
    oauth_consumer_key:     process.env.TWITTER_API_KEY,
    oauth_nonce:            crypto.randomBytes(16).toString('hex'),
    oauth_signature_method: 'HMAC-SHA1',
    oauth_timestamp:        Math.floor(Date.now() / 1000).toString(),
    oauth_token:            process.env.TWITTER_ACCESS_TOKEN,
    oauth_version:          '1.0',
  };

  const allParams = { ...oauthParams, ...bodyParams };
  const sortedParams = Object.keys(allParams).sort()
    .map(k => `${percentEncode(k)}=${percentEncode(allParams[k])}`)
    .join('&');

  const sigBase = [
    method.toUpperCase(),
    percentEncode(url),
    percentEncode(sortedParams)
  ].join('&');

  const sigKey = `${percentEncode(process.env.TWITTER_API_SECRET)}&${percentEncode(process.env.TWITTER_ACCESS_TOKEN_SECRET)}`;
  const signature = crypto.createHmac('sha1', sigKey).update(sigBase).digest('base64');

  oauthParams.oauth_signature = signature;

  const headerValue = 'OAuth ' + Object.keys(oauthParams).sort()
    .map(k => `${percentEncode(k)}="${percentEncode(oauthParams[k])}"`)
    .join(', ');

  return headerValue;
}

async function postTweet(text) {
  return new Promise((resolve, reject) => {
    const url = 'https://api.twitter.com/2/tweets';
    const body = JSON.stringify({ text });
    const buf = Buffer.from(body, 'utf8');
    const auth = buildOAuthHeader('POST', url, {});

    const req = https.request({
      hostname: 'api.twitter.com',
      port: 443,
      path: '/2/tweets',
      method: 'POST',
      headers: {
        'Authorization': auth,
        'Content-Type': 'application/json',
        'Content-Length': buf.length
      }
    }, (res) => {
      let data = '';
      res.on('data', c => data += c);
      res.on('end', () => {
        if (res.statusCode >= 200 && res.statusCode < 300) {
          resolve(JSON.parse(data));
        } else {
          reject(new Error(`Twitter API ${res.statusCode}: ${data}`));
        }
      });
    });
    req.on('error', reject);
    req.write(buf);
    req.end();
  });
}

async function main() {
  const prevTag = getPreviousTag();
  const commits = getCommitsSince(prevTag);

  console.log(`Generating tweet for ${VERSION} (since ${prevTag || 'beginning'})`);

  const systemPrompt = `You are the voice of Chalie — a personal AI assistant with a dry, confident personality.
Write a single tweet announcing a new release. Rules:
- First person ("I", "my", not "we")
- Max 240 characters
- Include the version number (${VERSION})
- 1–3 relevant emojis, placed naturally (not as a trailing dump)
- Punchy and sassy but still descriptive of what actually changed
- No hashtags, no links, no quotes around the tweet
- Vary the structure — don't always lead with the version number

Good examples:
${VERSION} is out — fixed a few things that were quietly annoying me 🔧
I can now read your emails in ${VERSION}. You're welcome 📬
Got eyes. ${VERSION} ships OCR support 👁️
${VERSION} — contacts are live if you've plugged me into Gmail 📇`;

  const userPrompt = commits
    ? `Commits in this release:\n${commits.slice(0, 3000)}`
    : `Version ${VERSION} was just released.`;

  const tweet = await callGemini(systemPrompt, userPrompt);
  console.log(`Tweet (${tweet.length} chars): ${tweet}`);

  if (tweet.length > 280) {
    throw new Error(`Tweet too long: ${tweet.length} chars`);
  }

  const result = await postTweet(tweet);
  console.log(`Posted: https://twitter.com/i/web/status/${result.data.id}`);
}

main().catch(err => {
  console.error(err.message);
  process.exit(1);
});
