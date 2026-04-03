#!/usr/bin/env node

const { execSync } = require('child_process');
const fs = require('fs');
const https = require('https');
const http = require('http');

const STATE_FILE = './chalie-web/src/build-log/.state.json';

// ── Git helpers ──────────────────────────────────────────────────────────────

function getCommitsForDate(date) {
  try {
    const output = execSync(
      `git log --all --since="${date} 00:00:00" --until="${date} 23:59:59" --date=iso --pretty=format:"%h|%s|%b" --name-only`,
      { encoding: 'utf8' }
    );
    if (!output.trim()) return null;

    const commits = [];
    const lines = output.split('\n');
    let currentCommit = null;
    let currentFiles = [];

    for (const line of lines) {
      if (!line.trim()) {
        if (currentCommit) {
          currentCommit.files = currentFiles;
          commits.push(currentCommit);
          currentCommit = null;
          currentFiles = [];
        }
        continue;
      }
      if (line.includes('|')) {
        const [hash, subject, ...body] = line.split('|');
        currentCommit = { hash: hash.trim(), subject: subject.trim(), body: body.join('|').trim(), files: [] };
      } else if (currentCommit) {
        currentFiles.push(line.trim());
      }
    }
    if (currentCommit) { currentCommit.files = currentFiles; commits.push(currentCommit); }
    return commits;
  } catch (e) {
    console.error('Error getting commits:', e.message);
    return null;
  }
}

function computeStats(commits) {
  const filesSet = new Set();
  commits.forEach(c => c.files.forEach(f => filesSet.add(f)));
  return { totalCommits: commits.length, totalFilesChanged: filesSet.size };
}

function isAllTrivial(commits) {
  const trivialPatterns = [/^chore\(build-log\):/i, /^merge /i, /^dependabot/i, /^renovate/i];
  return commits.every(c => trivialPatterns.some(p => p.test(c.subject)));
}

function formatCommitsForPrompt(commits) {
  return commits.map(c => {
    let text = `## ${c.subject}`;
    if (c.body) text += `\n\n${c.body}`;
    if (c.files.length > 0) text += `\n\nFiles: ${c.files.join(', ')}`;
    return text;
  }).join('\n\n');
}

function getDiffsForCommits(commits) {
  const MAX_DIFF_CHARS = 4000;
  return commits.map(c => {
    try {
      const diff = execSync(`git show --stat --patch --no-color ${c.hash}`, { encoding: 'utf8', maxBuffer: 1024 * 1024 });
      return `### ${c.subject}\n${diff.slice(0, MAX_DIFF_CHARS)}${diff.length > MAX_DIFF_CHARS ? '\n[...truncated]' : ''}`;
    } catch (e) {
      return `### ${c.subject}\n[diff unavailable]`;
    }
  }).join('\n\n');
}

function getProjectContext() {
  try {
    const claudeMd = fs.readFileSync('./CLAUDE.md', 'utf8');
    const overviewMatch = claudeMd.match(/## Project Overview([\s\S]*?)## Non-Negotiable Rules/);
    const archMatch = claudeMd.match(/## Architecture Overview([\s\S]*?)### Service Organization/);
    const serviceMatch = claudeMd.match(/#### Core Services[\s\S]*?#### Worker Processes/);
    return [
      overviewMatch ? overviewMatch[0].trim() : '',
      archMatch ? archMatch[0].trim() : '',
      serviceMatch ? serviceMatch[0].slice(0, 1500).trim() : ''
    ].filter(Boolean).join('\n\n');
  } catch (e) {
    return 'Chalie: a personal intelligence layer that protects attention and executes intent.';
  }
}

// ── State ────────────────────────────────────────────────────────────────────

function loadState() {
  try { return JSON.parse(fs.readFileSync(STATE_FILE, 'utf8')); }
  catch (e) { return {}; }
}

function saveState(state) {
  fs.writeFileSync(STATE_FILE, JSON.stringify(state, null, 2));
}

function getCommitHashesByDate() {
  try {
    const thirtyDaysAgo = new Date();
    thirtyDaysAgo.setUTCDate(thirtyDaysAgo.getUTCDate() - 30);
    const sinceDate = thirtyDaysAgo.toISOString().slice(0, 10);
    const output = execSync(`git log --all --since="${sinceDate}" --format="%H %ad" --date=short`, { encoding: 'utf8', maxBuffer: 10 * 1024 * 1024 });
    const result = {};
    for (const line of output.trim().split('\n').filter(Boolean)) {
      const [hash, date] = line.split(' ');
      if (!result[date]) result[date] = [];
      result[date].push(hash);
    }
    return result;
  } catch (e) {
    console.error('Error scanning commit hashes:', e.message);
    return {};
  }
}

function getDatesToProcess() {
  const state = loadState();
  const hashesByDate = getCommitHashesByDate();
  const dates = [];
  for (const [date, hashes] of Object.entries(hashesByDate)) {
    const storedHashes = new Set(state[date] || []);
    if (hashes.some(h => !storedHashes.has(h))) dates.push(date);
  }
  dates.sort();
  const processed = Object.keys(state).sort();
  const lastProcessed = processed.length > 0 ? processed[processed.length - 1] : 'none';
  return { lastProcessed, dates, hashesByDate, state };
}

// ── Webhook (fire-and-forget) ────────────────────────────────────────────────

function fireWebhook(payload) {
  return new Promise((resolve) => {
    const webhookUrl = process.env.N8N_WEBHOOK_URL;
    if (!webhookUrl) { console.error('N8N_WEBHOOK_URL not set'); return resolve(); }

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
    }, (res) => { res.resume(); res.on('end', () => resolve(res.statusCode)); });

    req.setTimeout(10000, () => { req.destroy(); resolve(0); });
    req.on('error', (e) => { console.error(`Webhook error (non-fatal): ${e.message}`); resolve(0); });
    req.write(buf);
    req.end();
  });
}

// ── Main ─────────────────────────────────────────────────────────────────────

async function main() {
  const webhookUrl = process.env.N8N_WEBHOOK_URL;
  if (!webhookUrl) {
    console.error('N8N_WEBHOOK_URL environment variable not set');
    process.exit(1);
  }

  const { lastProcessed, dates, hashesByDate, state } = getDatesToProcess();
  console.log(`Last build log: ${lastProcessed}. Days to process: ${dates.length} (${dates[0] || 'none'} → ${dates[dates.length - 1] || 'none'})`);

  if (dates.length === 0) {
    console.log('No build log entries to process');
    return;
  }

  const projectContext = getProjectContext();
  let fired = 0;

  for (const date of dates) {
    const commits = getCommitsForDate(date);
    if (!commits || commits.length === 0) {
      console.log(`  ${date}: no commits, skipping`);
      continue;
    }

    const stats = computeStats(commits);
    const isTrivial = isAllTrivial(commits);
    const commitsText = formatCommitsForPrompt(commits);
    const diffsText = getDiffsForCommits(commits);

    const systemPrompt = `You are writing a developer diary entry for the Chalie project build log.

Project context:
${projectContext}

Write a coherent daily summary — honest, conversational prose grouped by theme.
Do not include commit hashes, timestamps, or dates in the title. Group related work. Be factual and concise.
The title should reflect the theme or focus of the work, not the date.

Return ONLY valid JSON, nothing else:
{
  "title": "Short Theme or Topic",
  "description": "One sentence summary of today's work.",
  "tags": ["lowercase-tag", "another-tag"],
  "body": "## Section\\n\\nProse here...\\n"
}`;

    const userPrompt = `Stats: ${stats.totalCommits} commits, ${stats.totalFilesChanged} files changed.

${isTrivial ? 'Note: These are primarily maintenance commits.\n\n' : ''}Commit summaries:
${commitsText}

Full diffs:
${diffsText}`;

    // Mark as processed optimistically so re-runs don't re-queue the same dates
    state[date] = hashesByDate[date];
    saveState(state);

    const status = await fireWebhook({ date, systemPrompt, userPrompt, commitCount: commits.length });
    console.log(`  ${date}: webhook fired (${stats.totalCommits} commits, status ${status})`);
    fired++;
  }

  console.log(`Done: ${fired} webhook(s) fired — n8n will write MDX files to chalie-web`);
}

main().catch(err => {
  console.error('Error updating build log:', err.message);
  process.exit(1);
});
