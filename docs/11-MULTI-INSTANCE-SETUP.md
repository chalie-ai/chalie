# Multi-Instance Setup

Running multiple Chalie instances simultaneously on the same host.

## Overview

By default, each Chalie stack runs on the same Docker Compose project with fixed networking. To run multiple independent instances in parallel, configure the following:

- **Port isolation** — Each frontend listens on a different host port
- **Project naming** — Each stack has its own isolated Docker Compose project
- **Service naming** — Services resolve within their project's internal network

## Environment Configuration

### Quick Reference

Each instance needs its own `.env` file (or environment variables) with a unique `PORT`:

```bash
# Instance 1: ~/.env or pass to docker compose up
PORT=8081
POSTGRES_PASSWORD=chalie

# Instance 2: separate .env or pass to docker compose up
PORT=8082
POSTGRES_PASSWORD=chalie
```

### Full Environment Variables

```env
# External port for the web interface (http://localhost:<PORT>)
# Change this to run multiple Chalie instances on the same host.
PORT=8081

# PostgreSQL password
POSTGRES_PASSWORD=chalie

# Session cookie signing secret
# Generate: python3 -c "import secrets; print(secrets.token_hex(32))"
SESSION_SECRET_KEY=changeme-in-production-use-a-long-random-string

# Set to true when serving over HTTPS
COOKIE_SECURE=false
```

## Running Multiple Instances

### Using Environment Variables

```bash
# Instance 1 — home instance on port 8081
COMPOSE_PROJECT_NAME=chalie-home PORT=8081 docker compose up -d

# Instance 2 — work instance on port 8082
COMPOSE_PROJECT_NAME=chalie-work PORT=8082 docker compose up -d

# Instance 3 — separate project on port 8083
COMPOSE_PROJECT_NAME=chalie-demo PORT=8083 docker compose up -d
```

### Using Separate .env Files

Create separate `.env` files for each instance:

**~/.chalie-home/.env**
```env
COMPOSE_PROJECT_NAME=chalie-home
PORT=8081
POSTGRES_PASSWORD=chalie
SESSION_SECRET_KEY=<generate-a-unique-key>
COOKIE_SECURE=false
```

**~/.chalie-work/.env**
```env
COMPOSE_PROJECT_NAME=chalie-work
PORT=8082
POSTGRES_PASSWORD=chalie
SESSION_SECRET_KEY=<generate-a-unique-key>
COOKIE_SECURE=false
```

Then start each from its directory:

```bash
cd ~/.chalie-home
docker compose up -d

cd ~/.chalie-work
docker compose up -d
```

Or pass the `.env` file explicitly:

```bash
docker compose --env-file ~/.chalie-home/.env up -d
```

## Isolation

Each `COMPOSE_PROJECT_NAME` gets its own:

- **Database volume** — `chalie-home_postgres_data`, `chalie-work_postgres_data`, etc.
- **Redis volume** — `chalie-home_redis_data`, `chalie-work_redis_data`, etc.
- **Application volume** — `chalie-home_backend_data`, `chalie-work_backend_data`, etc.
- **Internal network** — `chalie-home_default`, `chalie-work_default`, etc.

Within each project's network:
- Services resolve by name: `postgres`, `redis`, `backend`, `frontend`
- No container name conflicts (automatic naming: `chalie-home-postgres-1`, `chalie-work-postgres-1`, etc.)
- PostgreSQL and Redis ports are NOT exposed to the host (for security and to prevent conflicts)

## Accessing Each Instance

```bash
# Instance 1
http://localhost:8081

# Instance 2
http://localhost:8082

# Instance 3
http://localhost:8083

# Brain dashboard for instance 1
http://localhost:8081/brain/

# Brain dashboard for instance 2
http://localhost:8082/brain/
```

## Database Access

Since PostgreSQL ports are not exposed to the host, use `docker compose exec`:

```bash
# Access PostgreSQL for chalie-home
docker compose -p chalie-home exec postgres psql -U postgres chalie

# Access PostgreSQL for chalie-work
docker compose -p chalie-work exec postgres psql -U postgres chalie

# Or via explicit env:
docker compose --env-file ~/.chalie-home/.env exec postgres psql -U postgres chalie
```

If you need direct host access (e.g., for backups or debugging), temporarily add ports back to `docker-compose.yml`:

```yaml
postgres:
  ports:
    - "5432:5432"  # chalie-home only
```

Then change for each project:

```yaml
postgres:
  ports:
    - "5432:5432"  # chalie-home
    - "5433:5432"  # chalie-work (map to different host port)
```

## Stopping & Cleaning Up

Stop a specific instance:

```bash
docker compose -p chalie-home down
docker compose -p chalie-work down
```

Or with explicit `.env`:

```bash
docker compose --env-file ~/.chalie-home/.env down
```

Stop all instances:

```bash
docker compose -p chalie-home down
docker compose -p chalie-work down
docker compose -p chalie-demo down
```

View all running instances:

```bash
docker ps --filter label=com.docker.compose.project=chalie-home
docker ps --filter label=com.docker.compose.project=chalie-work
docker ps | grep chalie
```

## Logs & Debugging

View logs for a specific instance:

```bash
docker compose -p chalie-home logs -f backend
docker compose -p chalie-work logs -f frontend
```

Or with explicit `.env`:

```bash
docker compose --env-file ~/.chalie-home/.env logs -f backend
```

## Verification Checklist

After starting a new instance:

1. ✅ Frontend loads: `http://localhost:<PORT>/`
2. ✅ Onboarding accessible: `http://localhost:<PORT>/on-boarding/`
3. ✅ Brain dashboard accessible: `http://localhost:<PORT>/brain/`
4. ✅ Database accessible: `docker compose -p <project> exec postgres psql -U postgres chalie`
5. ✅ Database is isolated (check `\l` — only `chalie` database should exist)
6. ✅ No port conflicts with other instances
7. ✅ No container name conflicts: `docker ps` shows `<project>-service-N` names

## Notes

- **Session secrets**: Each instance can share the same `SESSION_SECRET_KEY` or use unique keys. Using unique keys means sessions don't transfer between instances (recommended for security).
- **Encryption key**: If not set, each instance generates its own encryption key for API key storage. For data portability, set `DB_ENCRYPTION_KEY` to the same value across instances (or leave unset for isolation).
- **Resource limits**: Running multiple instances uses more CPU/memory. Monitor `docker stats` and adjust limits in `docker-compose.yml` if needed.
- **Firewall**: Ports are only exposed to localhost by default. Update docker-compose.yml if you need external access.
