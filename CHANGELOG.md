# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Persistent tasks system for long-running operations with state preservation.
- Plan decomposition capability breaking complex goals into actionable steps.
- Cognitive reflexes enabling immediate responses without full context loading.
- Document synthesis tool creating structured summaries from conversations.
- Camera OCR integration for extracting text from images and screenshots.
- Curiosity threads tracking surprising or interesting topics over time.
- Ambient awareness mode monitoring system state passively in background.
- Place learning feature remembering locations and spatial contexts.
- Wikipedia tool providing direct access to encyclopedia articles.
- Focus sessions allowing dedicated work periods with reduced interruptions.

### Changed
- Migrated from multi-process architecture to single Python process for simplicity.
- Made Docker optional instead of required for core functionality.
- Integrated voice native support directly into the main interface.
- Established trusted tool tier bypassing sandbox requirements for verified tools.

### Fixed
- Resolved documentation SEO metadata generation causing file corruption.
- Removed rogue scripts accidentally committed during cleanup phase.
- Reverted unintended global search-and-replace changes in markdown files.
