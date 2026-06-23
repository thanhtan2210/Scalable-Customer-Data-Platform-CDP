# Phase 0: Database Foundation & Persistence Infrastructure

**Date:** June 2026
**Scope:** Foundation & Infrastructure (Managed PostgreSQL)

This document defines the foundational infrastructure for data persistence using **Supabase (PostgreSQL)**. By using a managed database, we ensure high availability and eliminate the need to run heavy PostgreSQL Docker containers on resource-constrained local machines.

## 1. Core Objectives of Database Infrastructure

In Phase 0, the Database layer must provide a secure and scalable repository for metadata (dataset profiles, training job statuses, and user configurations).

- **Zero-Local Database:** No PostgreSQL server runs locally. All relational data is stored in the Cloud.
- **Connection Pooling:** Infrastructure must support high-concurrency requests from both the Backend API and background worker tasks.
- **Schema Evolution:** Managed via SQLAlchemy models and Alembic migrations (to be established in later phases).

## 2. Infrastructure Setup (Supabase)

We utilize **Supabase** because it provides a dedicated PostgreSQL instance on their Free Tier with a built-in UI for data exploration.

### Connection Parameters
- **Database URL:** `postgresql://postgres.[PROJECT_ID]:[PASSWORD]@aws-0-[REGION].pooler.supabase.com:5432/postgres`
- **Port:** 5432 (Standard) or 6543 (Transaction Pooler).
- **SSL:** Required for all remote connections.

## 3. Organizational Strategy

To ensure data integrity, the database uses a structured schema:

- **Metadata Tables:** Tracking `datasets` (ID, status, cloud_path), `profiles` (column stats, inferred roles), and `training_jobs` (MLflow run IDs, performance metrics).
- **Environment Isolation:** Different databases or schemas are used to separate `development` and `production` data.

## 4. Environment Configuration

The infrastructure is activated by injecting the `DATABASE_URL` into the environment:

```env
# Format: postgresql://user:password@host:port/dbname
DATABASE_URL=postgresql://postgres.xxx:pass@xxx.pooler.supabase.com:5432/postgres
```

## 5. Integration Strategy

The platform uses **SQLAlchemy** (Asynchronous) for the ORM layer.

**Key Lifecycle Foundation:**
1. **Engine Initialization:** Creating a thread-safe engine via `create_engine()` or `create_async_engine()`.
2. **Session Management:** Using a singleton `SessionLocal` factory to provide fresh database connections for each API request.
3. **Automatic Fallback:** If `DATABASE_URL` is missing or invalid, the system defaults to a local **SQLite** database (`sqlite:///./test.db`) for immediate testing.

## 6. Connectivity Security

- **IP Whitelisting:** If necessary, Supabase access should be restricted to the developer's IP.
- **SSL Verification:** Connections must enforce SSL to prevent man-in-the-middle attacks.
