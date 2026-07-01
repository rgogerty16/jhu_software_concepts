"""Generate module_6_report.pdf — run once to (re)build the submission report.

Requires fpdf2:  pip install fpdf2

Covers every topic the assignment's PDF must address:
  - Docker Compose architecture and the role of each service
  - RabbitMQ message flow (publisher -> exchange -> queue -> consumer)
  - Database initialisation and idempotent/incremental loading
  - Worker behaviour (task map, per-message transaction, ack/nack)
  - Verification steps, and the two required screenshots

Screenshots are embedded automatically when present:
  screenshots/website.png    - the running app at http://localhost:8080
  screenshots/rabbitmq.png   - the RabbitMQ management UI at http://localhost:15672
If a screenshot file is missing, a labelled placeholder is drawn instead, so the
report always builds; drop the PNGs in and re-run to finalise.
"""

import os

from fpdf import FPDF

TITLE = "Module 6 - Deploy Anywhere (Dockerised Microservices)"
AUTHOR = "Ryan Gogerty - JHU Modern Concepts in Python, Module 6"

# Replace with your Docker Hub namespace before submitting.
DOCKERHUB_USER = "rgogerty"

SCREENSHOTS = [
    ("screenshots/website.png",
     "Figure 1 - The running web app at http://localhost:8080 (Grad Cafe analytics)."),
    ("screenshots/rabbitmq.png",
     "Figure 2 - The RabbitMQ management UI at http://localhost:15672 (tasks_q queue)."),
]

SECTIONS = [
    ("1. Docker Compose Architecture",
     "The stack is defined entirely in docker-compose.yml and comes up on a clean "
     "machine with 'docker compose up --build'. Four services share Compose's "
     "default private network and address each other by service name. 'db' is a "
     "PostgreSQL 16 container with a named volume (pgdata) for durable storage and "
     "an init.sql bootstrap mounted into /docker-entrypoint-initdb.d. 'rabbitmq' is "
     "the message broker (RabbitMQ 3 with the management plugin), exposing AMQP on "
     "5672 and the management UI on 15672. 'web' is the Flask front end, built from "
     "src/web/Dockerfile and published on localhost:8080. 'worker' is the Python "
     "consumer, built from src/worker/Dockerfile, with the data directory bind-"
     "mounted read-only at /data. Both application images are built from the ./src "
     "context so they can share one 'db' package (a single connection helper and "
     "JSON loader), install only their own pinned requirements, and run as a non-"
     "root user (USER 1000). Health checks on db and rabbitmq gate the app "
     "containers via depends_on: condition: service_healthy, so nothing starts "
     "talking to a broker or database that is not ready yet."),

    ("2. Service Roles and Why the Tiers Are Split",
     "Decoupling the web tier from data-modifying work improves reliability, "
     "scalability, and security. The web app stays fast and stateless: it renders "
     "analytics and, when a button is clicked, publishes a task and immediately "
     "returns HTTP 202 - it never waits on a scrape or a heavy recompute. The "
     "worker owns all long-running and write operations, so it can be scaled, "
     "retried, and rate-limited independently. Routing every write through the "
     "queue gives buffering and backpressure (no request timeouts under load), "
     "durable at-least-once delivery, and clear operational visibility through the "
     "broker's queue metrics. This mirrors the classic producer/consumer and "
     "least-privilege patterns."),

    ("3. RabbitMQ Message Flow",
     "The web tier's publisher.py opens a channel (pika BlockingConnection using "
     "RABBITMQ_URL) and idempotently declares a durable direct exchange named "
     "'tasks', a durable queue named 'tasks_q', and a binding on routing key "
     "'tasks'. publish_task(kind, payload) sends a compact JSON body "
     "{kind, ts, payload} with delivery_mode=2 (a persistent message that survives "
     "a broker restart) and closes the connection in a finally block; any failure "
     "is raised so the Flask endpoint can return 503 rather than silently drop "
     "work. The Pull Data button publishes kind 'scrape_new_data'; Update Analysis "
     "publishes 'recompute_analytics'. The worker's consumer.py connects with the "
     "same URL, declares the same durable topology, and sets "
     "basic_qos(prefetch_count=1) so it processes exactly one message at a time - "
     "backpressure that prevents a slow task from letting the queue overwhelm the "
     "worker."),

    ("4. Database Initialisation, Migration, and SQL Safety",
     "Schema creation happens two ways, both idempotent. On a clean machine, "
     "Postgres runs src/db/init.sql from its init directory, creating the "
     "applicants table, the ingestion_watermarks table, and the analysis_summary "
     "cache. Independently, the worker calls db.load_data.create_schema() at start-"
     "up, so the schema also exists in non-Docker runs (local dev and CI). "
     "load_data.py then bulk-loads data/applicant_data.json with parameterised, "
     "sql.SQL-composed INSERTs and ON CONFLICT (url) DO NOTHING, so re-running "
     "never duplicates rows. Every statement across the codebase binds values "
     "through placeholders and quotes identifiers with sql.Identifier - runtime "
     "input can never alter a query's shape."),

    ("5. Worker Behaviour: Idempotent, Watermarked, Transactional",
     "consumer.on_message parses each JSON message and routes by 'kind' through a "
     "task map to handle_scrape_new_data or handle_recompute_analytics. Each "
     "message runs in its own transaction (a connection opened per message that "
     "commits only on clean exit); the message is acknowledged only after that "
     "commit. On any handler error the transaction rolls back and the message is "
     "basic_nack'd with requeue=False, so a malformed or poison task is dropped "
     "rather than looping forever. handle_scrape_new_data reads the high-water mark "
     "from ingestion_watermarks, fetches only records with a larger Grad Cafe "
     "result id, inserts them idempotently, and advances the watermark to the max "
     "id seen - so repeated Pull Data clicks add only genuinely new rows. "
     "handle_recompute_analytics recomputes the cached scalar metrics into "
     "analysis_summary; the web page shows those cached values and their 'last "
     "recomputed' timestamp, which is how the asynchronous recompute becomes "
     "visible in the UI."),

    ("6. Build, Run, and Verification Steps",
     "1) From module_6/, run 'docker compose up --build'. 2) Wait for the db and "
     "rabbitmq health checks to pass; the worker then seeds the base data and "
     "recomputes the cache. 3) Open http://localhost:8080 to see the analytics "
     "page. 4) Open http://localhost:15672 (guest/guest, dev only) to see the "
     "'tasks_q' queue. 5) Click Pull Data - the app returns a queued banner (202); "
     "watch a message flow through tasks_q and the live entry count rise after the "
     "worker commits. 6) Click Update Analysis - the cached summary and its "
     "timestamp refresh. The screenshots below capture the running website and the "
     "RabbitMQ management UI."),

    ("7. Container Registry (Docker Hub)",
     "Both images are published to a public Docker Hub repository named module_6. "
     "Build and push with, for example: "
     "'docker build -t " + DOCKERHUB_USER + "/module_6:web-v1 -f src/web/Dockerfile src' and "
     "'docker build -t " + DOCKERHUB_USER + "/module_6:worker-v1 -f src/worker/Dockerfile src', "
     "then 'docker login' and 'docker push " + DOCKERHUB_USER + "/module_6:web-v1' (and the "
     "worker tag). The public repository is at "
     "https://hub.docker.com/r/" + DOCKERHUB_USER + "/module_6, and pull/run "
     "instructions and the live registry link are in the README."),

    ("8. Quality Gates: Pylint and Pytest",
     "A dedicated Module 6 GitHub Actions workflow runs on changes under module_6/. "
     "The lint job installs the service requirements plus pylint and scores the "
     "source at 10.00/10 (--fail-under=10). The test job spins up a Postgres "
     "service container and runs pytest with --cov-fail-under=100 over src/, so the "
     "39-test suite must pass at 100% coverage. The Module 5 workflows are scoped to "
     "module_5/ paths and do not interfere."),
]


def _a(text):
    """Map common Unicode punctuation to ASCII so the latin-1 core font renders."""
    replacements = {
        "—": "-", "–": "-", "’": "'", "‘": "'",
        "“": '"', "”": '"', "…": "...", "→": "->",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


class PDF(FPDF):
    """Report document with a running header and page-number footer."""

    def header(self):
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(26, 39, 68)
        self.cell(0, 9, TITLE, align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 6, AUTHOR, align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(3)
        self.set_draw_color(200, 210, 230)
        self.set_line_width(0.5)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def _add_screenshot(pdf, path, caption):
    """Embed a screenshot if present, else draw a labelled placeholder box."""
    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_text_color(80, 80, 80)
    width = pdf.w - pdf.l_margin - pdf.r_margin
    if os.path.exists(path):
        pdf.image(path, w=width)
    else:
        start_y = pdf.get_y()
        pdf.set_draw_color(180, 180, 180)
        pdf.set_fill_color(245, 245, 248)
        pdf.rect(pdf.l_margin, start_y, width, 55, style="DF")
        pdf.set_xy(pdf.l_margin, start_y + 24)
        pdf.multi_cell(width, 6,
                       _a(f"[ Add screenshot: {path} ]"), align="C")
        pdf.set_y(start_y + 55)
    pdf.ln(2)
    pdf.multi_cell(0, 5, _a(caption))
    pdf.ln(5)


def build():
    """Render every section and screenshot into module_6_report.pdf."""
    pdf = PDF()
    pdf.set_margins(22, 22, 22)
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    for heading, body in SECTIONS:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(26, 39, 68)
        pdf.multi_cell(0, 7, _a(heading))
        pdf.ln(1)
        pdf.set_font("Helvetica", "", 10.5)
        pdf.set_text_color(30, 30, 30)
        pdf.multi_cell(0, 6, _a(body))
        pdf.ln(5)

    pdf.add_page()
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(26, 39, 68)
    pdf.multi_cell(0, 7, _a("9. Required Screenshots"))
    pdf.ln(3)
    for path, caption in SCREENSHOTS:
        _add_screenshot(pdf, path, caption)

    pdf.output("module_6_report.pdf")
    print("module_6_report.pdf written.")


if __name__ == "__main__":
    build()
