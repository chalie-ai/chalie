#!/usr/bin/env node

const { execSync } = require('child_process');
const fs = require('fs');
const https = require('https');

// Get today's date in UTC
const TODAY = new Date().toISOString().slice(0, 10);
const BUILD_LOG_DIR = './chalie-web/src/build-log';
const BUILD_LOG_FILE = `${BUILD_LOG_DIR}/${TODAY}.md`;

async function getCommitsForToday() {
  try {
    // Get all commits from today using UTC dates
    const output = execSync(
      `git log --since="${TODAY} 00:00:00" --until="${TODAY} 23:59:59" --date=iso --pretty=format:"%h|%s|%b" --name-only`,
      { encoding: 'utf8' }
    );

    if (!output.trim()) {
      console.log('No commits found for today');
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

async function callClaude(payload) {
  return new Promise((resolve, reject) => {
    const options = {
      hostname: 'api.anthropic.com',
      port: 443,
      path: '/v1/messages',
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': process.env.ANTHROPIC_API_KEY,
        'anthropic-version': '2023-06-01'
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
            resolve(JSON.parse(data));
          } catch (e) {
            reject(new Error(`Failed to parse response: ${e.message}`));
          }
        } else {
          reject(new Error(`API error ${res.statusCode}: ${data}`));
        }
      });
    });

    req.on('error', reject);
    req.write(JSON.stringify(payload));
    req.end();
  });
}

async function generateBuildLogEntry(commits, stats) {
  const isTrivial = isAllTrivial(commits);
  const commitsText = formatCommitsForPrompt(commits);

  const systemPrompt = `You are writing a developer diary entry for the Chalie project build log.
Write a coherent daily summary — honest, conversational prose grouped by theme.
Do not include commit hashes or timestamps. Group related work. Be factual and concise.

Return ONLY valid JSON, nothing else:
{
  "title": "Month Day — Short Theme or Topic",
  "description": "One sentence summary of today's work.",
  "tags": ["lowercase-tag", "another-tag"],
  "body": "## Section\n\nProse here...\n"
}`;

  const userPrompt = `Stats: ${stats.totalCommits} commits, ${stats.totalFilesChanged} files changed.

${isTrivial ? 'Note: These are primarily maintenance commits.\n\n' : ''}Today's commits:
${commitsText}`;

  const payload = {
    model: 'claude-haiku-4-5-20251001',
    max_tokens: 1024,
    system: systemPrompt,
    messages: [
      {
        role: 'user',
        content: userPrompt
      }
    ]
  };

  // Retry once on failure
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      const response = await callClaude(payload);
      const content = response.content[0].text;
      return JSON.parse(content);
    } catch (error) {
      if (attempt === 0) {
        console.log('Claude API call failed, retrying...');
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

function formatFrontmatter(entry, date) {
  const dateParts = date.split('-');
  const dateObj = new Date(`${date}T00:00:00Z`);
  const months = ['January', 'February', 'March', 'April', 'May', 'June',
                  'July', 'August', 'September', 'October', 'November', 'December'];
  const month = months[dateObj.getUTCMonth()];
  const day = dateObj.getUTCDate();

  const normalizedTags = normalizeTags(entry.tags);

  return `---
title: "${month} ${day} — ${entry.title}"
description: "${entry.description}"
date: ${date}
tags: [${normalizedTags.map(t => `"${t}"`).join(', ')}]
category: "Dev Log"
layout: build-log-post.njk
---

${entry.body}`;
}

async function main() {
  try {
    // Get commits for today
    const commits = await getCommitsForToday();

    if (!commits || commits.length === 0) {
      console.log('No commits to process');
      process.exit(0);
    }

    console.log(`Found ${commits.length} commits for ${TODAY}`);

    // Compute stats
    const stats = computeStats(commits);
    console.log(`Stats: ${stats.totalCommits} commits, ${stats.totalFilesChanged} files changed`);

    // Generate build log entry using Claude
    console.log('Calling Claude API to generate build log entry...');
    const entry = await generateBuildLogEntry(commits, stats);

    // Format the file content
    const fileContent = formatFrontmatter(entry, TODAY);

    // Ensure directory exists
    if (!fs.existsSync(BUILD_LOG_DIR)) {
      fs.mkdirSync(BUILD_LOG_DIR, { recursive: true });
    }

    // Write the file
    fs.writeFileSync(BUILD_LOG_FILE, fileContent);
    console.log(`Build log entry written to ${BUILD_LOG_FILE}`);

  } catch (error) {
    console.error('Error updating build log:', error.message);
    process.exit(1);
  }
}

main();
