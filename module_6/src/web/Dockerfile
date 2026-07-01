# Web service image — Flask front end + RabbitMQ publisher.
#
# Build context is ./src (see docker-compose.yml) so the shared `db` package can
# be copied alongside the web code and both services reuse one connection helper
# and one JSON loader. The image installs only pinned, service-specific deps and
# runs as a non-root user.
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so this layer caches across code changes.
COPY web/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code: web/ becomes /app (run.py, publisher.py, app/), plus the
# shared db package at /app/db.
COPY web/ ./
COPY db/ ./db/

EXPOSE 8080

# Run as an unprivileged user (least privilege). 1000 is the conventional
# first non-root uid and needs no write access to the read-only code.
USER 1000

CMD ["python", "run.py"]
