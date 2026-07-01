# Module 6 — Deploy Anywhere

**Name:** Ryan Gogerty
**JHED ID:** rgogerty
**Course:** Modern Concepts in Python
**Assignment:** Module 6 — Deploy Anywhere (Dockerised microservices with RabbitMQ)

The Module 5 Flask + PostgreSQL project, refactored into a four-service
microservice stack that runs anywhere via Docker Compose. The web app is
decoupled from all data-modifying work: button clicks publish tasks to RabbitMQ
and a background worker processes them.

---

## Architecture

```
                 publish task (202)            consume (prefetch=1)
   ┌────────┐  ─────────────────────►  ┌──────────┐  ─────────────►  ┌────────┐
   │  web   │        tasks / tasks_q   │ rabbitmq │                  │ worker │
   │ Flask  │                          │  broker  │                  │ python │
   │ :8080  │  ◄─── reads analytics ── │          │                  │  ETL   │
   └────┬───┘                          └──────────┘                  └───┬────┘
        │                                                                │
        │            ┌──────────────────────────┐                       │
        └──────────► │   db  (PostgreSQL 16)     │ ◄─────────────────────┘
           read      │   named volume: pgdata     │   write (idempotent)
                     └──────────────────────────┘
```

| Service    | Image                  | Role                                             | Port(s)        |
|------------|------------------------|--------------------------------------------------|----------------|
| `web`      | built (`src/web`)      | Flask UI + RabbitMQ **publisher**                | 8080           |
| `worker`   | built (`src/worker`)   | RabbitMQ **consumer** + ETL (scrape, recompute)  | —              |
| `db`       | `postgres:16`          | data store; schema auto-init; named volume       | (internal)     |
| `rabbitmq` | `rabbitmq:3-management`| message broker + management UI                   | 5672 / 15672   |

---

## Repository layout

```
module_6/
├── docker-compose.yml        # 4 services, healthchecks, named volume, private network
├── setup.py                  # packages the shared db library
├── pytest.ini / .pylintrc    # 100% coverage gate, 10/10 lint config
├── make_report_pdf.py        # regenerates module_6_report.pdf
├── docs/  tests/  screenshots/
└── src/
    ├── web/
    │   ├── Dockerfile  requirements.txt
    │   ├── run.py            # Flask entrypoint — binds 0.0.0.0:8080
    │   ├── publisher.py      # _open_channel() + publish_task()
    │   └── app/              # create_app factory, query_data, templates, static
    ├── worker/
    │   ├── Dockerfile  requirements.txt
    │   ├── consumer.py       # durable AMQP, prefetch=1, task map, ack/nack
    │   └── etl/
    │       ├── incremental_scraper.py   # watermark-driven ingestion
    │       └── query_data.py            # analytics recompute
    ├── db/
    │   ├── load_data.py      # schema + JSON loader + watermark/insert helpers
    │   ├── init.sql          # Docker auto-init schema
    │   └── db.py             # shared connection helper
    └── data/
        └── applicant_data.json          # LLM-cleaned data set (bind-mounted read-only)
```

> **Build context note:** both application images build from the `./src` context
> (`dockerfile: web/Dockerfile` / `worker/Dockerfile`) so they can share the one
> `db` package — a single connection helper and JSON loader — instead of
> duplicating it. Each image still installs only its own pinned `requirements.txt`
> and runs as a non-root user (`USER 1000`).

---

## Prerequisites

- **Docker Desktop / Engine** and the **Docker Compose** plugin.
  Verify with `docker --version` and `docker compose version`.

No local Python or Postgres is required to run the stack — everything runs in
containers.

---

## Quick start

```bash
cd module_6
docker compose up --build
```

Compose builds the images, starts Postgres and RabbitMQ, waits for their health
checks, then starts `web` and `worker`. The worker seeds the base data on
start-up and recomputes the cache.

| What                     | URL                              |
|--------------------------|----------------------------------|
| Web app (analytics)      | http://localhost:8080            |
| RabbitMQ management UI    | http://localhost:15672 (guest/guest — dev only) |

Tear down with `docker compose down` (add `-v` to also drop the `pgdata` volume).

### Task buttons

- **Pull Data** → `POST /pull-data` publishes `scrape_new_data` and returns **202**
  immediately. The worker ingests the next batch of new records (watermarked and
  idempotent); the live entry count rises after it commits.
- **Update Analysis** → `POST /update-analysis` publishes `recompute_analytics`
  and returns **202**. The worker refreshes the cached summary; the page's
  "Cached Summary" panel and its *last recomputed* timestamp update.

If the broker is unreachable, both endpoints return **503** (errors are surfaced,
never swallowed).

---

## Environment variables

Compose sets these for you (see `docker-compose.yml`); `.env.example` documents
them for local runs.

| Variable            | Used by      | Example                                                  |
|---------------------|--------------|----------------------------------------------------------|
| `DATABASE_URL`      | web, worker  | `postgresql://gradcafe:gradcafe@db:5432/gradcafe`        |
| `RABBITMQ_URL`      | web, worker  | `amqp://guest:guest@rabbitmq:5672/`                      |
| `DATA_FILE`         | worker       | `/data/applicant_data.json`                              |
| `INITIAL_LOAD_LIMIT`| worker       | `25000` (base slice; higher-id records feed the scraper) |
| `SCRAPE_BATCH`      | worker       | `500` (new records ingested per Pull Data click)         |

---

## Container registry (Docker Hub)

Public repository (must be named `module_6`):
**https://hub.docker.com/r/rgogerty/module_6**

Build, tag, and push both images:

```bash
# from module_6/
docker build -t rgogerty/module_6:web-v1    -f src/web/Dockerfile    src
docker build -t rgogerty/module_6:worker-v1 -f src/worker/Dockerfile src
docker login
docker push rgogerty/module_6:web-v1
docker push rgogerty/module_6:worker-v1
```

Pull and run the published images (Postgres + RabbitMQ still come from Compose):

```bash
docker pull rgogerty/module_6:web-v1
docker pull rgogerty/module_6:worker-v1
```

> Both images are published under the public `rgogerty` Docker Hub namespace; the
> `docker login` step authenticates as that account before pushing.

---

## Local development & tests

The suite runs against a local PostgreSQL and mocks RabbitMQ (no broker needed).

```bash
cd module_6
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r src/web/requirements.txt -r src/worker/requirements.txt pytest pytest-cov pylint

createdb gradcafe_m6_test                       # one-time local test DB
DATABASE_URL=postgresql://localhost/gradcafe_m6_test pytest    # 100% coverage gate

PYTHONPATH=src:src/web:src/worker pylint --rcfile=.pylintrc \
  src/db src/web/run.py src/web/publisher.py src/web/app \
  src/worker/consumer.py src/worker/etl                       # 10.00/10
```

`tests/conftest.py` puts `src`, `src/web`, and `src/worker` on `sys.path`, so
every service module imports in one process exactly as it does in its container.

---

## What changed from Module 5

| Topic            | Module 5                          | Module 6                                                    |
|------------------|-----------------------------------|-------------------------------------------------------------|
| Topology         | single Flask process              | 4 services (web / worker / db / rabbitmq) via Compose        |
| Long-running work| background thread in the web app  | RabbitMQ task → worker consumer (durable, prefetch=1)        |
| Button behaviour | blocks/polls scrape status        | publishes a task, returns **202** immediately               |
| Ingestion        | full re-scrape                    | **watermarked, incremental, idempotent** (`ingestion_watermarks`) |
| Analytics        | computed live only                | live **+** worker-refreshed `analysis_summary` cache         |
| Packaging/run    | `pip install -e .`                | Docker images (non-root), published to Docker Hub            |
| DB init          | manual `load_data.py`             | `init.sql` auto-init **+** worker start-up seed              |
