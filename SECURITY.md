# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability, please report it by emailing **security@chalie.ai**. We aim to respond within 48 hours and provide a fix within 7 days for critical issues. Please do not open public GitHub issues for security concerns.

## Security Design

### Local-First Architecture
Chalie is designed with privacy as the foundation:
- All data stored locally in SQLite database on user machine
- Zero telemetry or analytics sent to external servers
- No cloud dependencies required for core functionality
- User retains full control over all personal data and memory

### Data Protection
- Database encryption using `DB_ENCRYPTION_KEY` environment variable
- Session-based authentication with secure token generation
- Sensitive credentials stored only in environment variables, never in code
- Automatic session expiration after period of inactivity

### Tool Sandboxing
Tools are categorized by trust level:
- **Untrusted tools**: Execute in ephemeral Docker containers with network isolation
- **Trusted tools**: Run as subprocesses within the main process context
- All tool execution includes timeout limits and resource constraints
- Input validation performed before any external API calls

### Network Security
- ProxyFix middleware validates Host headers to prevent cache poisoning
- Default configuration binds only to localhost (127.0.0.1)
- CORS policies restrict cross-origin requests to configured origins
- HTTPS recommended for production deployments with valid certificates

## Supported Versions

| Version | Status   |
|---------|----------|
| Alpha   | ✅ Yes    |

Only the latest alpha version receives security updates and patches.

## Scope

### In Scope
- Authentication bypass vulnerabilities
- Data exfiltration through unauthorized access
- Remote code execution (RCE) via tool interfaces
- Sandbox escape from Docker containers or subprocess isolation
- SQL injection in database queries
- Cross-site scripting (XSS) in web interface

### Out of Scope
- Denial of Service (DoS) attacks against self-hosted instances
- Social engineering or phishing attempts
- Physical access to user machines
- Attacks requiring compromise of third-party services
- Issues related to outdated browser versions
