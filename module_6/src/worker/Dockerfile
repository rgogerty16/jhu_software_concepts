# Worker service image — long-running RabbitMQ consumer + ETL.
#
# Build context is ./src so the shared `db` package is copied next to the worker
# code (the worker reuses db.load_data for the base load and the watermark/insert
# helpers). Pinned deps, non-root user.
FROM python:3.11-slim

WORKDIR /app

COPY worker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code: worker/ becomes /app (consumer.py, etl/), plus the shared
# db package at /app/db. The data set is NOT baked in — it is bind-mounted
# read-only at /data by Docker Compose.
COPY worker/ ./
COPY db/ ./db/

USER 1000

CMD ["python", "consumer.py"]
