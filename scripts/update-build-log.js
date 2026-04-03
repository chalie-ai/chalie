#!/usr/bin/env node

const { execSync } = require('child_process');
const fs = require('fs');

// Get today's date in UTC
const TODAY = new Date().toISOString().slice(0, 10);
const BUILD_LOG_DIR = './chalie-web/src/build-log';


function getCommitsForDate(date) {
  try {
    const sinceDateStr = date + ' 00:00:00';
    const untilDateStr = date + ' 23:59:59';

    // Get commits for this specific date
    const output = execSync(
      `git log --all --since="${sinceDateStr}" --until="${untilDateStr}" --date=iso --pretty=format:"%h|%s|%b" --name-only`,
      { encoding: 'utf8' }
    );

    if (!output.trim()) {
      console.log(`No commits found for ${date}`);
      return null;
    }

    // Parse commits
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

      // Check if this is a commit line (contains |)
      if (line.includes('|')) {
        const [hash, subject, ...body] = line.split('|');
        currentCommit = {
          hash: hash.trim(),
          subject: subject.trim(),
          body: body.join('|').trim(),
          files: []
        };
      } else if (currentCommit) {
        // This is a file path
        currentFiles.push(line.trim());
      }
    }

    // Don't forget the last commit
    if (currentCommit) {
      currentCommit.files = currentFiles;
      commits.push(currentCommit);
    }

    return commits;
  } catch (error) {
    console.error('Error getting commits:', error.message);
    return null;
  }
}

function computeStats(commits) {
  const filesSet = new Set();
  commits.forEach(commit => {
    commit.files.forEach(file => filesSet.add(file));
  });

  return {
    totalCommits: commits.length,
    totalFilesChanged: filesSet.size
  };
}

function isAllTrivial(commits) {
  const trivialPatterns = [
    /^chore\(build-log\):/i,
    /^merge /i,
    /^dependabot/i,
    /^renovate/i
  ];

  return commits.every(commit =>
    trivialPatterns.some(pattern => pattern.test(commit.subject))
  );
}

function formatCommitsForPrompt(commits) {
  return commits
    .map(commit => {
      let text = `## ${commit.subject}`;
      if (commit.body) {
        text += `\n\n${commit.body}`;
      }
      if (commit.files.length > 0) {
        text += `\n\nFiles: ${commit.files.join(', ')}`;
      }
      return text;
    })
    .join('\n\n');
}

async function callGemini(systemPrompt, userPrompt) {
  const https = require('https');

  return new Promise((resolve, reject) => {
    const apiKey = process.env.GEMINI_API_KEY;
    const model = process.env.GEMINI_MODEL || 'gemini-2.0-flash';

    if (!apiKey) {
      return reject(new Error('GEMINI_API_KEY environment variable not set'));
    }

    const payload = JSON.stringify({
      contents: [
        {
          parts: [
            {
              text: userPrompt
            }
          ]
        }
      ],
      system_instruction: {
        parts: [
          {
            text: systemPrompt
          }
        ]
      }
    });

    const payloadBuffer = Buffer.from(payload, 'utf8');

    const options = {
      hostname: 'generativelanguage.googleapis.com',
      port: 443,
      path: `/v1beta/models/${model}:generateContent?key=${apiKey}`,
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': payloadBuffer.length
      }
    };

    const req = https.request(options, (res) => {
      let data = '';

      res.on('data', (chunk) => {
        data += chunk;
      });

      res.on('end', () => {
        if (res.statusCode >= 200 && res.statusCode < 300) {
          try {
            const response = JSON.parse(data);
            let text = response.candidates?.[0]?.content?.parts?.[0]?.text || '';
            if (!text) {
              return reject(new Error('No text in Gemini response'));
            }
            // Strip markdown code fences if present
            text = text.replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '').trim();
            const parsed = JSON.parse(text);
            resolve(parsed);
          } catch (e) {
            reject(new Error(`Failed to parse response: ${e.message}`));
          }
        } else {
          reject(new Error(`API error ${res.statusCode}: ${data}`));
        }
      });
    });

    req.on('error', reject);
    req.write(payloadBuffer);
    req.end();
  });
}

function getProjectContext() {
  try {
    const claudeMd = fs.readFileSync('./CLAUDE.md', 'utf8');
    // Extract the Project Overview and Architecture Overview sections for context
    const overviewMatch = claudeMd.match(/## Project Overview([\s\S]*?)## Non-Negotiable Rules/);
    const archMatch = claudeMd.match(/## Architecture Overview([\s\S]*?)### Service Organization/);
    const serviceMatch = claudeMd.match(/#### Core Services[\s\S]*?#### Worker Processes/);
    const parts = [
      overviewMatch ? overviewMatch[0].trim() : '',
      archMatch ? archMatch[0].trim() : '',
      serviceMatch ? serviceMatch[0].slice(0, 1500).trim() : ''
    ].filter(Boolean);
    return parts.join('\n\n');
  } catch (e) {
    return 'Chalie: a personal intelligence layer that protects attention and executes intent.';
  }
}

function getDiffsForCommits(commits) {
  const MAX_DIFF_CHARS = 4000;
  const diffs = [];
  for (const commit of commits) {
    try {
      const diff = execSync(
        `git show --stat --patch --no-color ${commit.hash}`,
        { encoding: 'utf8', maxBuffer: 1024 * 1024 }
      );
      // Trim per-commit diff to avoid overwhelming the prompt
      diffs.push(`### ${commit.subject}\n${diff.slice(0, MAX_DIFF_CHARS)}${diff.length > MAX_DIFF_CHARS ? '\n[...truncated]' : ''}`);
    } catch (e) {
      diffs.push(`### ${commit.subject}\n[diff unavailable]`);
    }
  }
  return diffs.join('\n\n');
}

async function generateBuildLogEntry(commits, stats) {
  const isTrivial = isAllTrivial(commits);
  const commitsText = formatCommitsForPrompt(commits);
  const projectContext = getProjectContext();
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
  "body": "## Section\n\nProse here...\n"
}`;

  const userPrompt = `Stats: ${stats.totalCommits} commits, ${stats.totalFilesChanged} files changed.

${isTrivial ? 'Note: These are primarily maintenance commits.\n\n' : ''}Commit summaries:
${commitsText}

Full diffs:
${diffsText}`;

  // Retry once on failure
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      const response = await callGemini(systemPrompt, userPrompt);
      return response;
    } catch (error) {
      if (attempt === 0) {
        console.log('Gemini API call failed, retrying...');
        // Wait a bit before retry
        await new Promise(resolve => setTimeout(resolve, 1000));
      } else {
        throw error;
      }
    }
  }
}

function normalizeTags(tags) {
  // Lowercase and deduplicate
  const normalized = tags.map(tag => tag.toLowerCase());
  return [...new Set(normalized)];
}

function formatFrontmatter(entry, date, commitCount) {
  const normalizedTags = normalizeTags(entry.tags);

  return `---
title: "${entry.title}"
description: "${entry.description}"
date: ${date}
commits: ${commitCount}
tags: [${normalizedTags.map(t => `"${t}"`).join(', ')}]
category: "Dev Log"
layout: build-log-post.njk
---

${entry.body}`;
}

const STATE_FILE = `${BUILD_LOG_DIR}/.state.json`;

// State schema: { "2026-04-01": ["abc123", "def456"], ... }
function loadState() {
  try {
    return JSON.parse(fs.readFileSync(STATE_FILE, 'utf8'));
  } catch (e) {
    return {};
  }
}

function saveState(state) {
  fs.writeFileSync(STATE_FILE, JSON.stringify(state, null, 2));
}

function getCommitHashesByDate() {
  // Returns { date: [hash, ...] } for last 30 days across all branches
  try {
    const thirtyDaysAgo = new Date();
    thirtyDaysAgo.setUTCDate(thirtyDaysAgo.getUTCDate() - 30);
    const sinceDate = thirtyDaysAgo.toISOString().slice(0, 10);
    const output = execSync(
      `git log --all --since="${sinceDate}" --format="%H %ad" --date=short`,
      { encoding: 'utf8', maxBuffer: 10 * 1024 * 1024 }
    );
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
    const hasNew = hashes.some(h => !storedHashes.has(h));
    if (hasNew) {
      dates.push(date);
    }
  }

  dates.sort();
  const processed = Object.keys(state).sort();
  const lastProcessed = processed.length > 0 ? processed[processed.length - 1] : 'none';
  return { lastProcessed, dates, hashesByDate, state };
}

async function main() {
  try {
    if (!fs.existsSync(BUILD_LOG_DIR)) {
      fs.mkdirSync(BUILD_LOG_DIR, { recursive: true });
    }

    const { lastProcessed, dates, hashesByDate, state } = getDatesToProcess();
    console.log(`Last build log: ${lastProcessed}. Days to process: ${dates.length} (${dates[0] || 'none'} → ${dates[dates.length - 1] || 'none'})`);

    let totalWritten = 0;
    for (const date of dates) {
      const commits = getCommitsForDate(date);
      if (!commits || commits.length === 0) {
        console.log(`  ${date}: no commits, skipping`);
        continue;
      }

      const stats = computeStats(commits);
      console.log(`  ${date}: ${stats.totalCommits} commits, ${stats.totalFilesChanged} files changed`);

      const entry = await generateBuildLogEntry(commits, stats);
      const fileContent = formatFrontmatter(entry, date, commits.length);
      const filePath = `${BUILD_LOG_DIR}/${date}.md`;
      fs.writeFileSync(filePath, fileContent);

      // Record the full set of hashes seen for this date so future runs skip it
      state[date] = hashesByDate[date];
      saveState(state);

      console.log(`  ${date}: written (${hashesByDate[date].length} hashes recorded)`);
      totalWritten++;
    }

    if (totalWritten === 0) {
      console.log('No build log entries to write');
    } else {
      console.log(`Done: ${totalWritten} build log entries written`);
    }

  } catch (error) {
    console.error('Error updating build log:', error.message);
    process.exit(1);
  }
}

main();
